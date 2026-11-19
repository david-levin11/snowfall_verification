#!/usr/bin/env python3
"""Compare QC snowfall reports with station-sampled NBM/NDFD archives.

Dependencies: numpy, pandas, pyarrow. See README.md and example_config.json.
All times are UTC; all comparison amounts are inches. No amount prorating.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def utc(value):
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise ValueError('Missing timestamp')
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def hours(delta):
    return delta.total_seconds() / 3600


def prepare(path):
    d = pd.read_csv(path)
    if 'report_id' not in d:
        d.insert(0, 'report_id', [f'R{i:04d}' for i in range(1, len(d)+1)])
    for c in ['obs_start', 'obs_end', 'archive_station_id', 'include', 'notes']:
        if c not in d:
            d[c] = 'yes' if c == 'include' else ''
        # Empty CSV columns otherwise infer float64; subsequent text assignment
        # warns in pandas 2 and fails in newer pandas releases.
        d[c] = d[c].astype('object')
    return d


def observations(args):
    d = prepare(args.obs)
    if d.report_id.isna().any() or d.report_id.duplicated().any():
        raise ValueError('report_id must be unique and nonmissing')
    d['observed_in'] = pd.to_numeric(d.snowfall, errors='coerce')
    if args.obs_units == 'cm':
        d['observed_in'] /= 2.54
    elif args.obs_units == 'mm':
        d['observed_in'] /= 25.4
    if getattr(args,'storm_total',False):
        # Preserve any supplied boundaries as metadata, but do not use them to
        # assign a time window to the QC'ed storm-total amount in this mode.
        d['reported_obs_start']=d['obs_start']
        d['reported_obs_end']=d['obs_end']
        d['obs_start']=pd.NaT
        d['obs_end']=pd.NaT
        d['window_basis']='qc_storm_total'
        return d
    for i, r in d.iterrows():
        missing_start, missing_end = pd.isna(r.obs_start), pd.isna(r.obs_end)
        missing_start |= str(r.obs_start).strip() == ''
        missing_end |= str(r.obs_end).strip() == ''
        if missing_start != missing_end:
            raise ValueError(f'{r.report_id}: provide BOTH obs_start and obs_end')
        assumed = missing_start and missing_end and args.assume_event_window
        d.loc[i, 'window_basis'] = 'assumed_event' if assumed else 'explicit'
        if assumed:
            d.loc[i, ['obs_start', 'obs_end']] = [args.event_start, args.event_end]
    return d


def distance(lat, lon, lats, lons):
    a, b = np.radians(float(lat)), np.radians(np.asarray(lats, dtype=float))
    dl = np.radians((np.asarray(lons, dtype=float)-float(lon)+180) % 360-180)
    h = np.sin((b-a)/2)**2 + np.cos(a)*np.cos(b)*np.sin(dl/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def load_metadata(path):
    d = pd.read_csv(path, dtype={'stid': str, 'station_id': str})
    d = d.rename(columns={'stid': 'station_id', 'Lat': 'latitude', 'Lon': 'longitude'})
    d = d[['station_id', 'latitude', 'longitude']].copy()
    d.station_id = d.station_id.str.strip().str.upper()
    for c in ['latitude', 'longitude']:
        d[c] = pd.to_numeric(d[c], errors='raise')
    if d.isna().any().any() or d.station_id.duplicated().any():
        raise ValueError('Station metadata must have unique IDs and valid coordinates')
    if (~d.latitude.between(-90, 90)).any() or (~d.longitude.between(-180, 360)).any():
        raise ValueError('Invalid metadata coordinates')
    return d.set_index('station_id')


def map_station(r, metadata, nearest_km):
    explicit = str(r.archive_station_id).strip().upper() if pd.notna(r.archive_station_id) else ''
    ident = explicit or str(r.get('stid', r.stationname)).strip().upper()
    try:
        lat, lon = float(r.Lat), float(r.Lon)
        valid = np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 360
    except (TypeError, ValueError):
        valid = False
    if not valid:
        return None, None, None, 'invalid_report_coordinates'
    if ident in metadata.index:
        s = metadata.loc[ident]
        return ident, float(distance(lat, lon, [s.latitude], [s.longitude])[0]), 'explicit' if explicit else 'id', None
    if explicit:
        return None, None, None, 'explicit_station_missing_from_metadata'
    if nearest_km is None:
        return None, None, None, 'no_station_id_match'
    ds = distance(lat, lon, metadata.latitude, metadata.longitude)
    if len(ds) == 0 or ds.min() > nearest_km:
        return None, None, None, 'no_station_within_distance'
    j = int(np.argmin(ds))
    return metadata.index[j], float(ds[j]), 'nearest', None


def selected_nbm_period(config, args):
    settings=config.get('verification',{})
    if not isinstance(settings,dict):raise ValueError('verification configuration must be an object')
    explicit=getattr(args,'nbm_accum_hours',None)
    period=explicit if explicit is not None else settings.get('nbm_accum_hours')
    if period is not None and period not in [6,24,48,72]:
        raise ValueError('nbm_accum_hours must be 6, 24, 48 or 72')
    return period


def load_sources(config, base, args):
    frames = []
    coverage = []
    nbm_period=selected_nbm_period(config,args)
    if nbm_period is not None:
        available=sorted({int(s['hours']) for s in config['sources'] if s['model'].lower()=='nbm'})
        if nbm_period not in available:
            raise ValueError(f'NBM {nbm_period}h was requested, but no matching source is configured. '
                             f'Configured NBM periods: {available}. Add that product to sources; no fallback is used.')
        print(f'NBM accumulation fixed at {nbm_period}h; other NBM accumulation products are skipped.')
    for source_number, s in enumerate(config['sources'], start=1):
        model = s['model'].lower()
        period = int(s['hours'])
        requested_init = getattr(args, model+'_init', None) or args.init
        if model not in ['nbm', 'ndfd'] or period not in [6, 24, 48, 72]:
            raise ValueError('Sources require model nbm/ndfd and hours 6/24/48/72')
        if model == 'ndfd' and period != 6:
            raise ValueError('NDFD inputs must be six-hour increments')
        if model=='nbm' and nbm_period is not None and period!=nbm_period:
            continue
        units = s['units']
        factor = {'in': 1, 'cm': 1/2.54, 'mm': 1/25.4, 'm': 39.3700787402}[units]
        patterns = s['files'] if isinstance(s['files'], list) else [s['files']]
        files = sorted({f for p in patterns for f in glob.glob(str(base / p), recursive=True)})
        if not files:
            raise ValueError(f'No files matched for {model} {period}h: {patterns}')
        if not s['columns']:
            raise ValueError('Each source needs at least one value column')
        for f in files:
            available = set(pq.read_schema(f).names)
            cols = ['station_id', 'valid_time', *s['columns']]
            derived_init = 'init_time' not in available
            if derived_init:
                if model != 'ndfd' or 'forecast_hour' not in available:
                    raise ValueError(f'{f}: missing init_time; NDFD may instead supply numeric forecast_hour')
                cols.append('forecast_hour')
            else:
                cols.append('init_time')
            missing = set(cols)-available
            if missing:
                raise ValueError(f'{f}: missing columns {sorted(missing)}; available columns: {sorted(available)}')
            d = pd.read_parquet(f, columns=list(dict.fromkeys(cols)))
            d['source_row'] = np.arange(1, len(d)+1)
            d['end'] = pd.to_datetime(d.valid_time, utc=True, errors='raise')
            if derived_init:
                lead = pd.to_numeric(d.forecast_hour, errors='raise')
                if lead.isna().any() or (~np.isfinite(lead)).any() or (lead < 0).any():
                    raise ValueError(f'{f}: forecast_hour must contain finite nonnegative hours')
                d['init_time'] = d.end-pd.to_timedelta(lead, unit='h')
            else:
                d['init_time'] = pd.to_datetime(d.init_time, utc=True, errors='raise')
            if d[['init_time', 'end', 'station_id']].isna().any().any():
                raise ValueError(f'Missing archive key in {f}')
            d['start'] = d.end-pd.Timedelta(hours=period)
            if (d.start < d.init_time).any():
                raise ValueError(f'{f}: interval begins before issuance; inspect archive timestamps')
            inventory = d.groupby('init_time', dropna=False).agg(
                n_rows=('station_id', 'size'), n_stations=('station_id', 'nunique'),
                first_window_start=('start', 'min'), last_window_end=('end', 'max'))
            inventory = inventory.reset_index()
            inventory['model'] = model
            inventory['period_h'] = period
            inventory['source_file'] = str(Path(f).resolve())
            inventory['requested_init'] = requested_init or ''
            inventory['selected'] = inventory.init_time.eq(utc(requested_init)) if requested_init else True
            coverage.append(inventory)
            if requested_init:
                d = d[d.init_time == utc(requested_init)]
            d['station_id'] = d.station_id.astype(str).str.strip().str.upper()
            d = d.melt(id_vars=['station_id', 'init_time', 'start', 'end', 'source_row'],
                       value_vars=s['columns'], var_name='field', value_name='forecast_in')
            d.forecast_in = pd.to_numeric(d.forecast_in, errors='raise')*factor
            if ((d.forecast_in < 0) | np.isinf(d.forecast_in)).any():
                raise ValueError(f'Negative or infinite snowfall in {f}')
            d['model'], d['period_h'] = model, period
            d['source_file'] = str(Path(f).resolve())
            d['source_entry'] = source_number
            d['init_time_basis'] = 'valid_time_minus_forecast_hour' if derived_init else 'init_time_column'
            frames.append(d)
    if not frames:
        raise ValueError('No sources configured')
    if getattr(args, 'output', None):
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(coverage, ignore_index=True).to_csv(out/'forecast_coverage.csv', index=False)
    inventory = pd.concat(coverage, ignore_index=True)
    for (model, period), part in inventory.groupby(['model', 'period_h']):
        chosen = part[part.selected]
        if chosen.empty:
            print(f'{model.upper()} {period}h: no records match requested initialization '
                  f'{part.requested_init.iloc[0]}. See forecast_coverage.csv for available runs.')
        else:
            print(f'{model.upper()} {period}h: loaded {chosen.init_time.nunique()} run(s). '
                  'Archive coverage is listed in forecast_coverage.csv; this is not the matched window.')
    d = pd.concat(frames, ignore_index=True)
    keys = ['model', 'station_id', 'init_time', 'start', 'end', 'field']
    policy = getattr(args, 'duplicate_policy', 'error')
    if policy not in ['error', 'last']:
        raise ValueError('duplicate_policy must be error or last')
    # Count NaN as a distinct value: a missing occurrence must not silently
    # replace a valid value unless the caller explicitly chooses last.
    conflicting = d.groupby(keys, dropna=False).forecast_in.transform('nunique', dropna=False) > 1
    audit = d.loc[conflicting].copy()
    report_path = None
    if getattr(args, 'output', None):
        report_path = Path(args.output)/'duplicate_forecasts.csv'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        audit['duplicate_policy'] = policy
        audit['kept'] = ~audit.duplicated(keys, keep='last') if policy == 'last' else False
        # Always write, including an empty header-only report, to avoid a stale audit.
        audit.to_csv(report_path, index=False)
    if not audit.empty:
        nkeys = len(audit[keys].drop_duplicates())
        detail = audit.groupby(['model', 'field']).size().to_dict()
        message = (f'Conflicting duplicate forecast keys: {nkeys} keys, {len(audit)} occurrences; '
                   f'counts by model/field: {detail}.')
        if report_path:
            message += f' Inspect {report_path.resolve()} for values and source row numbers.'
        if policy == 'error':
            raise ValueError(message + ' Use --duplicate-policy last only if last-occurrence selection is appropriate.')
        print(message + ' Keeping the last input occurrence per key; this does not prove it is the newest forecast.')
    # Preserve input order until selection, even for an A/B/A sequence. Removing
    # identical rows first with keep=first would incorrectly choose B with last.
    return d.drop_duplicates(keys, keep='last')


def windows(g, model, start, end, tolerance):
    """Yield amount-blind candidates. NDFD chains contain no missing increments."""
    if model == 'nbm':
        for r in g.itertuples():
            if pd.notna(r.forecast_in):
                yield r.start, r.end, r.forecast_in, 1
        return
    g = g.sort_values('end')
    records = list(g.itertuples())
    for i, first in enumerate(records):
        if abs(hours(first.start-start)) > tolerance:
            continue
        total, previous = 0.0, first.start
        for r in records[i:]:
            if r.start != previous or pd.isna(r.forecast_in):
                break
            total += r.forecast_in
            previous = r.end
            if abs(hours(r.end-end)) <= tolerance:
                yield first.start, r.end, total, int(round(hours(r.end-first.start)/6))
            if r.end > end+pd.Timedelta(hours=tolerance):
                break


def compare_storm_totals(obs, forecasts, metadata, args):
    """Compare QC event totals to an explicitly requested forecast window."""
    target_end=utc(args.forecast_end)
    nbm_hours=getattr(args,'nbm_accum_hours',None)
    ndfd_hours=getattr(args,'ndfd_accum_hours',None) or nbm_hours
    selected=[];rejected=[]
    for _,r in obs.iterrows():
        base=dict(r.to_dict(),obs_start=pd.NaT,obs_end=pd.NaT,window_basis='qc_storm_total')
        if str(r.include).lower().strip() not in ['yes','true','1']:
            rejected.append(dict(base,reason='excluded_by_user'));continue
        if not np.isfinite(r.observed_in) or r.observed_in<0:
            rejected.append(dict(base,reason='invalid_observed_amount'));continue
        sid,km,method,reason=map_station(r,metadata,args.nearest_km)
        base.update(matched_station_id=sid,distance_km=km,spatial_method=method)
        if reason:
            rejected.append(dict(base,reason=reason));continue
        f=forecasts[forecasts.station_id.eq(sid)]
        if f.empty:
            rejected.append(dict(base,reason='station_has_no_selected_forecasts'));continue
        for (model,init,field),g in f.groupby(['model','init_time','field']):
            duration=nbm_hours if model=='nbm' else ndfd_hours
            if duration is None:raise ValueError(f'Storm-total mode requires an explicit accumulation period for {model.upper()}')
            start=target_end-pd.Timedelta(hours=duration)
            info=dict(base,model=model,init_time=init,field=field,
                      requested_forecast_start=start,requested_forecast_end=target_end)
            if init>start:
                rejected.append(dict(info,reason='issuance_after_requested_forecast_start'));continue
            matches=[(fs,fe,value,n) for fs,fe,value,n in windows(g,model,start,target_end,0)
                     if fs==start and fe==target_end]
            if not matches:
                rejected.append(dict(info,reason='requested_forecast_window_unavailable'));continue
            if len(matches)!=1:raise ValueError('More than one forecast matches the explicit storm-total window')
            fs,fe,value,n=matches[0]
            selected.append(dict(info,forecast_start=fs,forecast_end=fe,forecast_duration_h=duration,
                observation_duration_h=np.nan,start_offset_h=np.nan,end_offset_h=np.nan,
                duration_difference_h=np.nan,temporal_overlap_fraction=np.nan,
                forecast_in=value,error_in=value-r.observed_in,absolute_error_in=abs(value-r.observed_in),
                n_increments=n,timing_class='storm_total_period_unspecified',timing_cost_h=np.nan,
                forecast_window_match='exact_requested_window',lead_to_start_h=hours(fs-init),
                selected=True,candidate_rank=1))
    return pd.DataFrame(selected),pd.DataFrame(rejected)


def compare(obs, forecasts, metadata, args):
    if getattr(args,'storm_total',False):
        return compare_storm_totals(obs,forecasts,metadata,args)
    candidates, rejected = [], []
    for _, r in obs.iterrows():
        base = r.to_dict()
        reason = None
        if str(r.include).lower().strip() not in ['yes', 'true', '1']:
            reason = 'excluded_by_user'
        elif not np.isfinite(r.observed_in) or r.observed_in < 0:
            reason = 'invalid_observed_amount'
        elif pd.isna(r.obs_start) or pd.isna(r.obs_end) or not str(r.obs_start).strip() or not str(r.obs_end).strip():
            reason = 'missing_observation_window'
        if reason:
            rejected.append({**base, 'reason': reason})
            continue
        start, end = utc(r.obs_start), utc(r.obs_end)
        if end <= start:
            raise ValueError(f'{r.report_id}: obs_end must follow obs_start')
        sid, km, method, reason = map_station(r, metadata, args.nearest_km)
        base.update(obs_start=start, obs_end=end, matched_station_id=sid, distance_km=km, spatial_method=method)
        if reason:
            rejected.append({**base, 'reason': reason})
            continue
        f = forecasts[forecasts.station_id == sid]
        if f.empty:
            rejected.append({**base, 'reason': 'station_has_no_selected_forecasts'})
            continue
        for (model, init, field), g in f.groupby(['model', 'init_time', 'field']):
            info = dict(base, model=model, init_time=init, field=field)
            if init > start:
                rejected.append(dict(info, reason='issuance_after_observation_start'))
                continue
            found = []
            for fs, fe, value, n in windows(g, model, start, end, args.endpoint_tolerance_h):
                ds, de = hours(fs-start), hours(fe-end)
                duration_difference = hours(fe-fs)-hours(end-start)
                overlap = max(0, hours(min(fe, end)-max(fs, start)))
                union = hours(max(fe, end)-min(fs, start))
                if max(abs(ds), abs(de)) > args.endpoint_tolerance_h or overlap <= 0:
                    continue
                if args.duration_tolerance_h is not None and abs(duration_difference) > args.duration_tolerance_h:
                    continue
                found.append(dict(info, forecast_start=fs, forecast_end=fe,
                                  forecast_duration_h=hours(fe-fs), observation_duration_h=hours(end-start),
                                  start_offset_h=ds, end_offset_h=de, duration_difference_h=duration_difference,
                                  temporal_overlap_fraction=overlap/union, forecast_in=value,
                                  error_in=value-r.observed_in, absolute_error_in=abs(value-r.observed_in),
                                  n_increments=n, timing_class='exact' if ds == de == 0 else 'approximate',
                                  timing_cost_h=abs(ds)+abs(de), lead_to_start_h=hours(fs-init)))
            if not found:
                rejected.append(dict(info, reason='no_complete_window_within_tolerances'))
                continue
            # Rank on timing only, never on the observed amount or forecast error.
            found.sort(key=lambda x: (x['timing_cost_h'], abs(x['duration_difference_h']), x['forecast_start'], x['forecast_end']))
            for i, row in enumerate(found):
                row['selected'] = i == 0
                row['candidate_rank'] = i+1
                candidates.append(row)
    return pd.DataFrame(candidates), pd.DataFrame(rejected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obs', required=True)
    p.add_argument('--prepare', help='Write editable report window/station mapping CSV and exit')
    p.add_argument('--config')
    p.add_argument('--stations', help='Station metadata CSV: stid or station_id, latitude, longitude')
    p.add_argument('--output', default='snowfall_results')
    p.add_argument('--init', help='Exact issuance/init time, UTC; omit to retain each issuance separately')
    p.add_argument('--nbm-init', help='Exact NBM initialization, overriding --init for NBM only (typically 01/07/13/19Z)')
    p.add_argument('--nbm-accum-hours',type=int,choices=[6,24,48,72],
                   help='Use only this native NBM snowfall product; overrides verification.nbm_accum_hours in config')
    p.add_argument('--storm-total',action='store_true',help='Treat observations as QC storm totals with unspecified accumulation boundaries')
    p.add_argument('--forecast-end',help='Exact forecast accumulation endpoint in UTC, required with --storm-total')
    p.add_argument('--ndfd-accum-hours',type=int,help='NDFD total duration (positive multiple of 6); in storm-total mode defaults to the selected NBM period')
    p.add_argument('--ndfd-init', help='Exact NDFD reference time, overriding --init for NDFD only')
    p.add_argument('--obs-units', choices=['in', 'cm', 'mm'], default='in')
    p.add_argument('--assume-event-window', action='store_true')
    p.add_argument('--event-start')
    p.add_argument('--event-end')
    p.add_argument('--endpoint-tolerance-h', '--endpoint-tolerance', type=float, default=0)
    p.add_argument('--duration-tolerance-h', '--duration-tolerance', type=float)
    p.add_argument('--duplicate-policy', choices=['error', 'last'], default='error',
                   help='Conflicting forecast values: stop (default), or keep last input occurrence with audit CSV')
    p.add_argument('--nearest-km', type=float, help='Opt in to nearest archived station within this radius')
    p.add_argument('--plots', action='store_true', help='Create charts, maps and individual report graphics after verification (requires matplotlib)')
    p.add_argument('--plot-no-individual', action='store_true', help='Skip individual report graphics')
    p.add_argument('--plot-extent', nargs=4, type=float, metavar=('WEST','EAST','SOUTH','NORTH'),
                   help='Optional map lon/lat bounds; default automatically fits selected reports')
    p.add_argument('--plot-view',help='Named view from plotting.map_views in the verification config')
    p.add_argument('--plot-formats', nargs='+', choices=['png','pdf'], default=['png'])
    p.add_argument('--plot-rows-per-page',type=int,default=10)
    p.add_argument('--plot-percentiles',nargs='+',type=float,help='Optional subset for NBM maps/scatterplots; default plots all available percentiles')
    p.add_argument('--plot-cmap',help='Map colormap override; otherwise use plotting.error_cmap')
    p.add_argument('--plot-error-range',nargs=2,type=float,metavar=('MIN','MAX'))
    p.add_argument('--plot-error-step',type=float)
    p.add_argument('--plot-error-levels',nargs='+',type=float)
    args = p.parse_args()
    if args.prepare:
        prepare(args.obs).to_csv(args.prepare, index=False)
        print(f'Prepared {args.prepare}; fill obs_start/obs_end and station mappings as needed.')
        return
    if not args.config or not args.stations:
        p.error('--config and --stations are required for comparison')
    if args.assume_event_window and (not args.event_start or not args.event_end):
        p.error('Event assumption requires --event-start and --event-end')
    for name in ['endpoint_tolerance_h', 'duration_tolerance_h', 'nearest_km']:
        v = getattr(args, name)
        if v is not None and (not np.isfinite(v) or v < 0):
            p.error(f'{name} must be finite and nonnegative')
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    args.nbm_accum_hours=selected_nbm_period(config,args)
    if args.storm_total:
        if args.assume_event_window or args.event_start or args.event_end:
            p.error('--storm-total uses --forecast-end; omit event-start, event-end and assume-event-window')
        if not args.forecast_end:p.error('--storm-total requires --forecast-end')
        utc(args.forecast_end)
        models={s['model'].lower() for s in config['sources']}
        if 'nbm' in models and args.nbm_accum_hours is None:
            p.error('--storm-total requires --nbm-accum-hours (or verification.nbm_accum_hours in config)')
        if args.ndfd_accum_hours is None:args.ndfd_accum_hours=args.nbm_accum_hours
        if 'ndfd' in models and (args.ndfd_accum_hours is None or args.ndfd_accum_hours<=0 or args.ndfd_accum_hours%6):
            p.error('NDFD storm-total duration must be a positive multiple of 6; set --ndfd-accum-hours')
        print('Observation mode: QC storm totals; report valid dates are metadata, not accumulation endpoints. '
              'Event-window and duration tolerances are not used; requested forecast windows must match exactly.')
        target_end=utc(args.forecast_end)
        for model in sorted(models):
            duration=args.nbm_accum_hours if model=='nbm' else args.ndfd_accum_hours
            target_start=target_end-pd.Timedelta(hours=duration)
            print(f'Requested {model.upper()} forecast window: {target_start} to {target_end} ({duration}h).')
            issue=getattr(args,model+'_init',None) or args.init
            if issue and utc(issue)>target_start:
                print(f'{model.upper()}: requested window starts before selected initialization {utc(issue)}. '
                      'This run cannot supply the full requested accumulation; use an earlier run or a later endpoint.')
    elif args.forecast_end or args.ndfd_accum_hours is not None:
        p.error('--forecast-end and --ndfd-accum-hours require --storm-total')
    obs = observations(args)
    forecasts = load_sources(config, config_path.parent, args)
    candidates, unmatched = compare(obs, forecasts, load_metadata(args.stations), args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out/'candidates.csv', index=False)
    unmatched.to_csv(out/'unmatched.csv', index=False)
    selected = candidates[candidates.selected].copy() if not candidates.empty else candidates
    selected.to_csv(out/'comparisons.csv', index=False)
    for model in sorted({s['model'].lower() for s in config['sources']}):
        part=selected[selected.model.eq(model)] if not selected.empty else selected
        if part.empty:
            print(f'{model.upper()}: 0 matched comparisons; inspect unmatched.csv for rejection reasons.')
            continue
        print(f'{model.upper()}: {len(part)} matched comparison rows from {part.report_id.nunique()} reports.')
        for r in part[['init_time','forecast_start','forecast_end','forecast_duration_h']].drop_duplicates().itertuples():
            print(f'  Matched forecast: init {r.init_time}; {r.forecast_start} to {r.forecast_end} ({r.forecast_duration_h:g}h).')
    if not selected.empty:
        # Multiple reports at one archive point are not independent. Expose counts.
        keys = ['model', 'init_time', 'field', 'timing_class', 'window_basis', 'ObType']
        summary = selected.groupby(keys, dropna=False).agg(
            n_reports=('report_id', 'size'), n_archive_stations=('matched_station_id', 'nunique'),
            mean_error_in=('error_in', 'mean'), mae_in=('absolute_error_in', 'mean'))
        summary.to_csv(out/'summary.csv')
    else:
        pd.DataFrame(columns=['n_reports', 'n_archive_stations', 'mean_error_in', 'mae_in']).to_csv(out/'summary.csv', index=False)
    (out/'run_settings.json').write_text(json.dumps({'arguments': vars(args), 'sources': config}, indent=2))
    print(f'{len(selected)} selected comparisons; {len(unmatched)} unmatched/excluded report-series records. Output: {out}')
    if args.plots:
        from plot_snowfall import plot_results
        plot_results(out/'comparisons.csv', out/'plots', individual=not args.plot_no_individual,
                     extent=args.plot_extent, formats=args.plot_formats,
                     rows_per_page=args.plot_rows_per_page,map_percentiles=args.plot_percentiles,
                     config=config,map_view=args.plot_view,map_cmap=args.plot_cmap,
                     error_range=args.plot_error_range,error_step=args.plot_error_step,error_levels=args.plot_error_levels)


if __name__ == '__main__':
    main()
