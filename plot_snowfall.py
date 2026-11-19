#!/usr/bin/env python3
"""Plot snowfall comparisons without reopening model archives.

Requires numpy, pandas, matplotlib. Coastlines are bundled (Natural Earth,
public domain); maps use a spherical north polar stereographic projection.
"""
import argparse
import hashlib
import html
import json
import math
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm, ListedColormap
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator

COLORS = {'nbm': '#7356a6', 'ndfd': '#df7c28'}
MARKERS = {'LSR': 'o', 'COOP': 's', 'CoCoRahs': 'D', 'SNOWDEPTH_SITE': '^'}
TIME_COLUMNS = ['init_time', 'forecast_start', 'forecast_end', 'obs_start', 'obs_end']
GROUP = ['report_id', 'model', 'init_time', 'forecast_start', 'forecast_end']


def stamp(value):
    if pd.isna(value):return 'unspecified'
    return pd.Timestamp(value).strftime('%b %d %H:%MZ')


def observation_caption(r):
    if r.window_basis=='qc_storm_total':
        return "Observed: QC'ed storm total (accumulation boundaries unspecified)"
    return f'Observed: {stamp(r.obs_start)} to {stamp(r.obs_end)}'


def set_timeline_ticks(ax, series):
    """Use explicit UTC ticks so common 24/48/72h windows do not trigger
    AutoDateLocator interval warnings, including single-model timelines."""
    times=[]
    for s in series:
        r=s['row']
        times.extend(pd.Timestamp(r[c]) for c in ['forecast_start','forecast_end','obs_start','obs_end'] if pd.notna(r[c]))
    low,high=min(times),max(times)
    span=max((high-low).total_seconds()/3600,1/60)
    intervals=[1/60,5/60,.25,.5,1,2,3,6,12,24,48,72,168,336,720,1440,8760]
    step=next((x for x in intervals if x>=span/4),span/4)
    ticks=list(pd.date_range(low,high,freq=pd.Timedelta(hours=step)))
    if not ticks or ticks[-1]!=high:ticks.append(high)
    ax.xaxis.set_major_locator(FixedLocator(mdates.date2num(ticks)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%MZ',tz='UTC'))


def slug(value):
    text = str(value)
    readable = re.sub(r'[^A-Za-z0-9_-]+', '_', text).strip('_')[:65] or 'report'
    return readable+'_'+hashlib.sha256(text.encode()).hexdigest()[:8]


def load_comparisons(path):
    try:
        d = pd.read_csv(path, dtype={'report_id': str, 'stationname': str, 'matched_station_id': str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if d.empty:
        return d
    required = GROUP+['stationname', 'observed_in', 'forecast_in', 'field', 'Lat', 'Lon',
                      'obs_start', 'obs_end', 'ObType', 'window_basis']
    missing = set(required)-set(d)
    if missing:
        raise ValueError(f'Comparison CSV lacks {sorted(missing)}')
    if 'selected' in d:
        d = d[d.selected.astype(str).str.lower().isin(['true', '1'])].copy()
    for c in TIME_COLUMNS:
        d[c] = pd.to_datetime(d[c], utc=True, errors='raise')
    unknown_period=d.window_basis.eq('qc_storm_total')
    if d[GROUP].isna().any().any() or (d[['obs_start','obs_end']].isna().any(axis=1)&~unknown_period).any():
        raise ValueError('Missing report/model/window identifiers in comparisons')
    for c in ['observed_in', 'forecast_in', 'Lat', 'Lon']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    if (~np.isfinite(d.observed_in) | ~np.isfinite(d.forecast_in)).any():
        raise ValueError('Comparison amounts must be finite')
    if d.duplicated(GROUP+['field']).any():
        raise ValueError('Duplicate selected report/model/window/field rows in plot input')
    # One report ID must continue to mean one observation, not multiple cases.
    identity = ['stationname', 'observed_in', 'obs_start', 'obs_end', 'ObType', 'Lat', 'Lon']
    if (d.groupby('report_id')[identity].nunique(dropna=False)>1).any().any():
        raise ValueError('A report_id has inconsistent observation metadata')
    d['model'] = d.model.astype(str).str.lower()
    return d


def series_rows(d):
    """Keep quantiles together ONLY within one report, run and native window."""
    result = []
    for _, group in d.groupby(GROUP, sort=False, dropna=False):
        percentiles = {}
        for r in group.itertuples():
            m = re.fullmatch(r'snow_p(\d+(?:\.\d+)?)', str(r.field))
            if m:
                percentiles[float(m[1])] = float(r.forecast_in)
            else:
                result.append(dict(row=group.loc[r.Index], quantiles={}, value=float(r.forecast_in), field=r.field))
        if percentiles:
            values = [percentiles[k] for k in sorted(percentiles)]
            result.append(dict(row=group.iloc[0], quantiles=percentiles,
                               value=percentiles.get(50), field='NBM quantiles',
                               crossed=bool(np.any(np.diff(values)<0))))
    return result


def series_label(s, window=False):
    r = s['row']
    duration=(r.forecast_end-r.forecast_start).total_seconds()/3600
    label = f"{r['model'].upper()} {duration:g}h | init {stamp(r.init_time)}"
    if not s['quantiles']:
        label += f" | {value_label(r)}"
    if window:
        label += f"\n{stamp(r.forecast_start)} to {stamp(r.forecast_end)}"
    return label


def value_label(r):
    m=re.fullmatch(r'snow_p(\d+(?:\.\d+)?)',str(r.field))
    if m:return 'p'+m[1]
    if r['model']=='ndfd' and pd.notna(r.get('n_increments')) and r.n_increments>1:
        return f'sum of {int(r.n_increments)} six-hour increments'
    return str(r.field)


def draw_amount(ax, s, y):
    r, q = s['row'], s['quantiles']
    color = COLORS.get(r['model'], '#247b83')
    crossed = s.get('crossed', False)
    if q:
        if crossed:
            ax.scatter(list(q.values()), [y]*len(q), marker='x', color=color, s=30, zorder=3)
            ax.text(max(q.values()), y, '  crossed quantiles', fontsize=7, color='#a22a36', va='center')
        else:
            if 10 in q and 90 in q:
                ax.plot([q[10], q[90]], [y, y], color=color, lw=2, zorder=2)
                ax.plot([q[10], q[90]], [y, y], '|', color=color, ms=9)
            if 25 in q and 75 in q:
                ax.plot([q[25], q[75]], [y, y], color=color, lw=9, solid_capstyle='butt', alpha=.55, zorder=2)
            for p, v in q.items():
                if p not in [10,25,50,75,90]:
                    ax.plot(v, y, '|', color=color, ms=9)
        if s['value'] is not None:
            ax.plot(s['value'], y, 'o', color=color, mec='white', mew=.6, ms=7, zorder=4)
        elif q:
            ax.text(max(q.values()), y+.18, 'p50 unavailable', fontsize=7, color=color)
    else:
        ax.plot(s['value'], y, 'D', color=color, mec='white', ms=7, zorder=4)
    ax.plot(r.observed_in, y, '*', color='#151c28', ms=12, zorder=5)


def style_amount(ax):
    ax.set_xlabel('Snowfall (inches)')
    ax.set_xlim(left=-.3)
    ax.grid(axis='x', color='#d9dfe5', lw=.7)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)


def amount_handles():
    return [Line2D([],[],marker='*',ls='',color='#151c28',ms=11,label='Observed'),
                       Line2D([],[],marker='o',ls='',color=COLORS['nbm'],label='NBM p50'),
                       Line2D([],[],lw=8,alpha=.55,color=COLORS['nbm'],label='NBM p25–p75'),
                       Line2D([],[],lw=2,color=COLORS['nbm'],label='NBM p10–p90'),
                       Line2D([],[],marker='D',ls='',color=COLORS['ndfd'],label='NDFD')]


def legend_amount(ax, compact=False):
    ax.legend(handles=amount_handles(),
              loc='upper left',bbox_to_anchor=(0,1.18 if compact else 1.065),
              ncol=3 if compact else 5,fontsize=8,frameon=False)


def paginate_series(series, limit):
    """Keep a report's forecasts together when they fit on a page."""
    pages=[];page=[]
    ids=list(dict.fromkeys(s['row'].report_id for s in series))
    for rid in ids:
        items=[s for s in series if s['row'].report_id==rid]
        if page and len(page)+len(items)>limit:
            pages.append(page);page=[]
        while len(items)>limit:
            pages.append(items[:limit]);items=items[limit:]
        page.extend(items)
    if page:pages.append(page)
    return pages


def overview_figure(part, title, page):
    n=len(part);height=max(7,n*.85+2.7)
    fig=plt.figure(figsize=(16,height))
    # Fixed physical space for title, subtitle and legend, independent of rows.
    ax=fig.add_axes([.38,.85/height,.60,1-2.75/height])
    labels=[]
    for i,s in enumerate(part):
        r=s['row'];draw_amount(ax,s,i)
        duration=(r.forecast_end-r.forecast_start).total_seconds()/3600
        labels.append(f"{r.stationname} [{r.report_id}] · {r.ObType}\n"
                      f"{r['model'].upper()} {duration:g}h | init {stamp(r.init_time)}\n"
                      f"{stamp(r.forecast_start)} to {stamp(r.forecast_end)}")
    ax.set_yticks(range(n),labels,fontsize=11)
    ax.set_ylim(n-.45,-.65);style_amount(ax)
    ax.tick_params(axis='x',labelsize=12);ax.xaxis.label.set_size(12)
    observed={(stamp(s['row'].obs_start),stamp(s['row'].obs_end)) for s in part}
    storm_total=all(s['row'].window_basis=='qc_storm_total' for s in part)
    subtitle=(observation_caption(part[0]['row']) if storm_total else
              'Observed: '+' to '.join(next(iter(observed))) if len(observed)==1 else
              'Multiple observation windows; see individual report graphics')
    fig.text(.025,1-.22/height,f'{title} | station comparisons {page}',fontsize=17,fontweight='bold',va='top')
    fig.text(.025,1-.62/height,subtitle+' · All times UTC',fontsize=12,va='top')
    fig.legend(handles=amount_handles(),loc='upper left',bbox_to_anchor=(.38,1-1.02/height),
               ncol=5,fontsize=11,frameon=False,borderaxespad=0)
    fig.text(.025,.16/height,'Star: observed. Each row is one model/run/window; forecasts for a report stay together where possible.',fontsize=10)
    return fig


def percentile_diagnostics(d):
    """Event-level quantile coverage and pinball loss; retain ties and sources."""
    q=d.copy()
    q['percentile']=pd.to_numeric(q.field.astype(str).str.extract(r'^snow_p(\d+(?:\.\d+)?)$')[0],errors='coerce')
    q=q[q.percentile.between(0,100,inclusive='neither')]
    keys=['model','init_time','forecast_start','forecast_end','obs_start','obs_end','ObType','window_basis','percentile']
    rows=[]
    for values,g in q.groupby(keys,sort=True,dropna=False):
        tau=g.percentile.iloc[0]/100
        error=g.observed_in-g.forecast_in
        row=dict(zip(keys,values))
        row.update(n_reports=g.report_id.nunique(),
                   n_archive_stations=g.matched_station_id.nunique() if 'matched_station_id' in g else np.nan,
                   fraction_observed_below=float((g.observed_in<g.forecast_in).mean()),
                   fraction_observed_at_or_below=float((g.observed_in<=g.forecast_in).mean()),
                   mean_pinball_loss_in=float(np.maximum(tau*error,(tau-1)*error).mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def circular_center(longitudes):
    lon = np.sort(np.asarray(longitudes, dtype=float)%360)
    if not len(lon):
        return -150.
    gaps = np.diff(np.r_[lon,lon[0]+360])
    i = int(np.argmax(gaps))
    west = lon[(i+1)%len(lon)]
    span = 360-gaps[i]
    return float((west+span/2+180)%360-180)


def project(lon, lat, center):
    # Spherical north polar stereographic, true scale at 60N, km.
    rho = 6371.0088*(1+np.sin(np.radians(60)))*np.tan(np.pi/4-np.radians(np.asarray(lat))/2)
    angle = np.radians((np.asarray(lon)-center+180)%360-180)
    return rho*np.sin(angle), -rho*np.cos(angle)


def validate_extent(extent):
    if not isinstance(extent,(list,tuple)) or len(extent)!=4:
        raise ValueError('Map extent must be [west, east, south, north]')
    try:
        west,east,south,north=map(float,extent)
    except (TypeError,ValueError) as exc:
        raise ValueError('Map extent coordinates must be numeric') from exc
    if not (np.isfinite([west,east,south,north]).all() and
            -180<=west<=180 and -180<=east<=180 and 0<south<north<=90):
        raise ValueError('Map extent needs west/east -180..180 and 0 < south < north <= 90')
    if (east-west)%360==0:
        raise ValueError('Map extent longitude span must be nonzero')
    return [west,east,south,north]


def resolve_map_view(config=None, extent=None, view=None):
    """Explicit CLI bounds > selected named view > config bounds > automatic."""
    if extent is not None:return validate_extent(extent),'custom'
    settings=(config or {}).get('plotting',{})
    if not isinstance(settings,dict):raise ValueError('plotting configuration must be an object')
    chosen=view if view is not None else settings.get('map_view')
    if chosen:
        views=settings.get('map_views',{})
        if not isinstance(views,dict):raise ValueError('plotting.map_views must be an object')
        if chosen not in views:
            raise ValueError(f'Unknown map view {chosen!r}; available views: {", ".join(views) or "none"}')
        return validate_extent(views[chosen]),chosen
    bounds=settings.get('map_extent')
    return (validate_extent(bounds),'custom') if bounds is not None else (None,'automatic')


def error_color_scale(config=None, *, error_limit=None, error_range=None,
                      error_step=None, error_levels=None, map_cmap=None):
    """Discrete anomaly colors centered on zero, including asymmetric ranges."""
    settings=(config or {}).get('plotting',{})
    if not isinstance(settings,dict):raise ValueError('plotting configuration must be an object')
    name=map_cmap or settings.get('error_cmap','RdBu_r')
    try:base=plt.get_cmap(name)
    except (ValueError,TypeError) as exc:raise ValueError(f'Unknown Matplotlib error colormap: {name!r}') from exc
    if error_limit is not None and error_range is not None:
        raise ValueError('Choose --error-limit or --error-range, not both')
    override_range=error_limit is not None or error_range is not None or error_step is not None
    levels=error_levels if error_levels is not None else None if override_range else settings.get('error_levels')
    if levels is None:
        bounds=error_range if error_range is not None else settings.get('error_range',[-10,10])
        if error_limit is not None:
            limit=float(error_limit)
            if not np.isfinite(limit) or limit<=0:raise ValueError('error_limit must be finite and positive')
            bounds=[-limit,limit]
        if not isinstance(bounds,(list,tuple)) or len(bounds)!=2:
            raise ValueError('error_range must be [minimum, maximum]')
        low,high=map(float,bounds)
        step=float(error_step if error_step is not None else settings.get('error_step',2))
        if not (np.isfinite([low,high,step]).all() and low<0<high and step>0):
            raise ValueError('Error range must span zero and error_step must be finite and positive')
        if (high-low)/step>200:raise ValueError('Too many error color bins; increase error_step')
        levels=np.unique(np.round(np.r_[np.arange(low,high,step),0.0,high],12)).tolist()
    try:levels=np.asarray(levels,dtype=float)
    except (TypeError,ValueError) as exc:raise ValueError('error_levels must be numeric boundaries') from exc
    if (levels.ndim!=1 or len(levels)<3 or len(levels)>201 or not np.isfinite(levels).all()
            or not np.all(np.diff(levels)>0) or not levels[0]<0<levels[-1]):
        raise ValueError('error_levels need 3–201 finite, increasing boundaries spanning zero')
    midpoint=(levels[:-1]+levels[1:])/2
    # Select negative/positive colors from the corresponding half of the palette
    # even if custom boundaries are asymmetric around zero.
    centered=TwoSlopeNorm(vmin=levels[0],vcenter=0,vmax=levels[-1])
    cmap=ListedColormap(base(centered(midpoint)),name='snowfall_error_bins')
    cmap.set_under(base(0.0));cmap.set_over(base(1.0))
    norm=BoundaryNorm(levels,ncolors=cmap.N,clip=False)
    return cmap,norm,dict(colormap=name,boundaries=levels.tolist())


def map_domain(d, extent=None, padding=30, minimum=100):
    valid = np.isfinite(d.Lat)&np.isfinite(d.Lon)&d.Lat.between(0.01,90)&d.Lon.between(-180,360)
    locations = d.loc[valid,['Lat','Lon']].drop_duplicates()
    if locations.empty:
        return None
    if extent:
        west,east,south,north = validate_extent(extent)
        span = (east-west)%360
        if span == 0:
            raise ValueError('Map extent longitude span must be nonzero')
        center = (west+span/2+180)%360-180
        lon = np.linspace(west,west+span,181)
        lat = np.linspace(south,north,181)
        px,py = project(np.r_[lon,lon,np.repeat(west,181),np.repeat(west+span,181)],
                        np.r_[np.repeat(south,181),np.repeat(north,181),lat,lat], center)
    else:
        center = circular_center(locations.Lon)
        px,py = project(locations.Lon,locations.Lat,center)
    xmid,ymid = (min(px)+max(px))/2,(min(py)+max(py))/2
    width,height = max(max(px)-min(px)+2*padding,minimum),max(max(py)-min(py)+2*padding,minimum)
    # Narrow coastal chains/Aleutian selections need enough surrounding context
    # for geography, legends and a scale bar; avoid a thin letterbox map.
    height=max(height,width/1.4)
    width=max(width,height/1.4)
    # Keep small station clusters legible and use the same bounds for every model.
    return dict(center=center,xlim=(xmid-width/2,xmid+width/2),ylim=(ymid-height/2,ymid+height/2))


@lru_cache(maxsize=1)
def land_rings():
    path = Path(__file__).with_name('ne_50m_land.geojson')
    if not path.exists():
        raise FileNotFoundError(f'Missing bundled coastline file: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    result=[]
    for feature in data['features']:
        g = feature['geometry']
        polygons = [g['coordinates']] if g['type']=='Polygon' else g['coordinates']
        for polygon in polygons:
            ring = np.asarray(polygon[0],dtype=float)
            # Include northern land only; clipping is visual, data are unchanged.
            if ring[:,1].max()>40:
                result.append(ring)
    return result


def draw_basemap(ax, domain):
    center = domain['center']
    ax.set_facecolor('#eaf1f6')
    for ring in land_rings():
        # Clamp Antarctic/far-southern points to avoid the south-pole singularity.
        x,y = project(ring[:,0],np.maximum(ring[:,1],-80),center)
        ax.fill(x,y,facecolor='#f4f3ed',edgecolor='#89959c',lw=.6,zorder=0)
    for lat in range(45,90,5):
        lons = np.linspace(center-180,center+180,721)
        x,y = project(lons,np.repeat(lat,len(lons)),center)
        ax.plot(x,y,color='#aebdc7',lw=.45,ls=':',zorder=1)
    for lon in range(-180,180,10):
        lats = np.linspace(40,89.95,200)
        x,y = project(np.repeat(lon,len(lats)),lats,center)
        ax.plot(x,y,color='#aebdc7',lw=.45,ls=':',zorder=1)
    ax.set(xlim=domain['xlim'],ylim=domain['ylim'],aspect='equal')
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_edgecolor('#aab6bf')
    ax.text(.01,.01,'North polar stereographic · Natural Earth 1:50m',transform=ax.transAxes,
            fontsize=7,color='#52606d',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    # A local distance bar in projected km (approximately ground distance near 60N).
    width = domain['xlim'][1]-domain['xlim'][0]
    length = 10**np.floor(np.log10(width*.2))
    if length*5<width*.25:length*=5
    elif length*2<width*.25:length*=2
    x0 = domain['xlim'][0]+width*.06
    y0 = domain['ylim'][0]+(domain['ylim'][1]-domain['ylim'][0])*.07
    ax.plot([x0,x0+length],[y0,y0],color='#33404b',lw=2)
    ax.text(x0+length/2,y0+(domain['ylim'][1]-domain['ylim'][0])*.015,
            f'~{length:g} km',ha='center',fontsize=8)


def plot_results(comparisons, output, *, individual=True, rows_per_page=10,
                 extent=None, labels=False, stations=None, report_ids=None,
                 map_percentile=None, map_percentiles=None, error_limit=None, formats=('png',), title='Snowfall event verification',
                 config=None, map_view=None, map_cmap=None, error_range=None, error_step=None, error_levels=None):
    extent,view_name=resolve_map_view(config,extent,map_view)
    error_cmap,color_norm,color_settings=error_color_scale(config,error_limit=error_limit,error_range=error_range,
        error_step=error_step,error_levels=error_levels,map_cmap=map_cmap)
    if rows_per_page<1:raise ValueError('rows_per_page must be positive')
    if not formats or any(f not in ['png','pdf'] for f in formats):raise ValueError('Formats must be png/pdf')
    if error_limit is not None and (not np.isfinite(error_limit) or error_limit<=0):
        raise ValueError('error_limit must be finite and positive')
    if map_percentile is not None and map_percentiles is not None:
        raise ValueError('Choose map_percentile or map_percentiles, not both')
    chosen_levels=[map_percentile] if map_percentile is not None else map_percentiles
    if chosen_levels is not None and (not len(chosen_levels) or any(not np.isfinite(x) or not 0<x<100 for x in chosen_levels)):
        raise ValueError('Map percentiles must be between 0 and 100')
    d = load_comparisons(comparisons)
    out = Path(output);out.mkdir(parents=True,exist_ok=True)
    if stations and not d.empty:
        wanted={s.upper() for s in stations}
        mask=d.stationname.str.upper().isin(wanted)
        if 'matched_station_id' in d:mask|=d.matched_station_id.str.upper().isin(wanted)
        d=d[mask]
    if report_ids and not d.empty:d=d[d.report_id.isin(report_ids)]
    manifest=[];warnings=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'savefig.facecolor':'white','axes.labelcolor':'#263340'})
    def save(fig, stem, kind, caption, report_id=''):
        paths=[]
        for fmt in formats:
            dest=out/(stem+'.'+fmt);dest.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(dest,dpi=160,bbox_inches='tight');paths.append(dest.relative_to(out).as_posix())
        plt.close(fig)
        manifest.append(dict(kind=kind,caption=caption,report_id=report_id,files=paths))
    if d.empty:
        warnings.append('No selected comparisons match the plot filters.')
    else:
        if d.window_basis.eq('assumed_event').any():
            warnings.append('Some observation windows were assumed from the event dates. Review the original reports.')
        if d.window_basis.eq('qc_storm_total').any():
            warnings.append("QC'ed storm totals are compared with explicitly selected forecast windows. Observation accumulation boundaries are unspecified; report valid dates are not treated as accumulation endpoints.")
        if d.ObType.eq('SNOWDEPTH_SITE').any():
            warnings.append('Triangles on maps/scatterplots represent snow-depth change, not a direct new-snow measurement.')
        warnings.append('Maps use report coordinates. Nearby archive-point distances are listed in individual report graphics.')
        if extent is not None:
            warnings.append(f'Map view: {view_name}. Zoom changes the map view only; charts and percentile statistics still use all selected reports, and map colors retain the configured range. Filter stations/reports to change the verification sample.')
        order=d[['report_id','observed_in']].drop_duplicates().sort_values(['observed_in','report_id'],ascending=[False,True]).report_id.tolist()
        rank={r:i for i,r in enumerate(order)}
        series=series_rows(d)
        series.sort(key=lambda s:(rank[s['row'].report_id],s['row']['model'],s['row'].init_time,s['row'].forecast_start,s['field']))
        if any(s.get('crossed') for s in series):warnings.append('Crossed NBM quantiles are drawn as crosses without filled intervals.')
        # Each series is its own row: multiple issuances/windows are never collapsed.
        for page,part in enumerate(paginate_series(series,rows_per_page),start=1):
            fig=overview_figure(part,title,page)
            save(fig,f'station_comparison_{page:03d}','overview',f'Station comparison page {page}')
        if individual:
            for rid in order:
                items=[s for s in series if s['row'].report_id==rid]
                for page,start in enumerate(range(0,len(items),rows_per_page),start=1):
                    part=items[start:start+rows_per_page];n=len(part);r=part[0]['row']
                    fig,(ax,tax)=plt.subplots(1,2,figsize=(14,max(5,n*.9+3)),gridspec_kw={'width_ratios':[1,1.1]})
                    for i,s in enumerate(part):
                        draw_amount(ax,s,i)
                        rr=s['row'];tax.plot([rr.forecast_start,rr.forecast_end],[i,i],lw=8,color=COLORS.get(rr['model'],'#247b83'),alpha=.65)
                        if pd.notna(rr.obs_start) and pd.notna(rr.obs_end):
                            tax.plot([rr.obs_start,rr.obs_end],[i+.22,i+.22],lw=3,color='#151c28')
                    ax.set_yticks(range(n),[series_label(s) for s in part],fontsize=8)
                    ax.set_ylim(n-.4,-.65);style_amount(ax);legend_amount(ax,compact=True)
                    tax.set_ylim(n-.4,-.65);tax.set_yticks([]);tax.grid(axis='x',alpha=.25)
                    set_timeline_ticks(tax,part)
                    tax.set_xlabel('Accumulation windows (UTC)')
                    timeline_title=("Forecast windows only\nObservation: QC'ed storm total; period unspecified"
                                    if r.window_basis=='qc_storm_total' else 'Color: forecast window · black: observation')
                    tax.set_title(timeline_title,fontsize=9)
                    flags=f"{r.ObType} | {r.window_basis} | observed {r.observed_in:g} in"
                    if 'matched_station_id' in r:
                        flags+=f" | archive point {r.matched_station_id}"
                    if 'distance_km' in r and pd.notna(r.distance_km):flags+=f" ({r.distance_km:.2f} km)"
                    fig.suptitle(f'{r.stationname} [{rid}]\n{flags}',fontsize=13,fontweight='bold',y=.99)
                    text=observation_caption(r)+'. '
                    text+='Forecast issue times and accumulation windows remain distinct.'
                    if 'notes' in r and pd.notna(r.notes):text+=' QC: '+str(r.notes)
                    fig.text(.01,.01,text,fontsize=8,wrap=True)
                    fig.subplots_adjust(left=.25,right=.98,bottom=.18,top=.72,wspace=.18)
                    save(fig,f'stations/{slug(rid+"_"+str(r.stationname))}_{page:02d}','station',str(r.stationname),rid)
        # All archived percentiles by default; explicit selection never falls back.
        quantile=d.field.astype(str).str.fullmatch(r'snow_p\d+(?:\.\d+)?')
        levels=pd.to_numeric(d.field.astype(str).str.extract(r'^snow_p(\d+(?:\.\d+)?)$')[0],errors='coerce')
        selected_level=levels.notna() if chosen_levels is None else levels.isin(chosen_levels)
        points=d[~quantile | selected_level].copy()
        if not quantile.any():
            warnings.append('No NBM percentile comparisons are present in this CSV. Check the NBM source columns and --nbm-init, then rerun verification.')
        elif chosen_levels is not None:
            for level in chosen_levels:
                if not levels.eq(level).any():warnings.append(f'No p{level:g} rows: that percentile is skipped; no substitute is used.')
        for model,g in d.groupby('model'):
            print(f'Plot input: {model.upper()} | {g.report_id.nunique()} reports | fields: {", ".join(sorted(g.field.unique()))}')
        domain=map_domain(d,extent)
        bad_coords=~(np.isfinite(d.Lat)&np.isfinite(d.Lon)&d.Lat.between(.01,90)&d.Lon.between(-180,360))
        if bad_coords.any():warnings.append(f'{d.loc[bad_coords,"report_id"].nunique()} reports have invalid map coordinates; their charts remain available.')
        print(f'Map colors: {color_settings["colormap"]} | boundaries: {color_settings["boundaries"]}')
        gkeys=['model','init_time','field','forecast_start','forecast_end','obs_start','obs_end']
        for number,(_,group) in enumerate(points.groupby(gkeys,sort=True,dropna=False),start=1):
            r=group.iloc[0];name=f"{r['model'].upper()} {value_label(r)} | init {stamp(r.init_time)}"
            windows=f"Forecast: {stamp(r.forecast_start)} – {stamp(r.forecast_end)}\n{observation_caption(r)}"
            if r.window_basis!='qc_storm_total' and (r.forecast_start!=r.obs_start or r.forecast_end!=r.obs_end):windows+=' | DIFFERENT WINDOWS'
            types=group.ObType.fillna('Unknown').unique()
            if domain is not None:
                fig,ax=plt.subplots(figsize=(10,8));draw_basemap(ax,domain)
                handles=[]
                for typ in types:
                    part=group[group.ObType.fillna('Unknown').eq(typ)]
                    valid=np.isfinite(part.Lat)&np.isfinite(part.Lon)&part.Lat.between(.01,90)&part.Lon.between(-180,360)
                    part=part[valid]
                    x,y=project(part.Lon,part.Lat,domain['center'])
                    sc=ax.scatter(x,y,c=part.forecast_in-part.observed_in,cmap=error_cmap,norm=color_norm,
                                  marker=MARKERS.get(typ,'o'),s=65,edgecolor='#354553',linewidth=.6,zorder=4)
                    handles.append(Line2D([],[],ls='',marker=MARKERS.get(typ,'o'),mfc='#ddd',mec='#354553',label=typ))
                    if labels:
                        for xx,yy,rr in zip(x,y,part.itertuples()):ax.annotate(rr.report_id,(xx,yy),xytext=(4,4),textcoords='offset points',fontsize=7,zorder=5)
                ax.legend(handles=handles,fontsize=8,loc='upper left')
                # Shared scale across all selected models and runs.
                bounds=color_settings['boundaries']
                tick_stride=max(1,math.ceil((len(bounds)-1)/12))
                ticks=sorted(set(bounds[::tick_stride]+[bounds[-1]]+([0] if 0 in bounds else [])))
                cb=fig.colorbar(plt.cm.ScalarMappable(norm=color_norm,cmap=error_cmap),ax=ax,orientation='horizontal',pad=.045,
                               shrink=.75,extend='both',boundaries=bounds,ticks=ticks,spacing='proportional',drawedges=True)
                cb.ax.tick_params(labelsize=8)
                cb.dividers.set_linewidth(.5)
                cb.dividers.set_color('#666666')
                cb.set_label('Forecast − observed snowfall (inches) · negative: too low · positive: too high',fontsize=9)
                map_name=(view_name+' | '+name) if view_name not in ['automatic','custom'] else name
                ax.set_title(f'{map_name}\n{windows}',fontsize=10,pad=12)
                save(fig,f'error_map_{number:03d}','map',map_name+' | '+windows.replace('\n','; '))
            fig,ax=plt.subplots(figsize=(8,7))
            for typ in types:
                part=group[group.ObType.fillna('Unknown').eq(typ)]
                ax.scatter(part.observed_in,part.forecast_in,marker=MARKERS.get(typ,'o'),s=50,
                           color=COLORS.get(r['model'],'#247b83'),edgecolor='white',lw=.5,label=typ,alpha=.8)
                if labels:
                    for rr in part.itertuples():ax.annotate(rr.report_id,(rr.observed_in,rr.forecast_in),xytext=(3,3),textcoords='offset points',fontsize=7)
            high=max(float(group.observed_in.max()),float(group.forecast_in.max()),1)*1.08
            ax.plot([0,high],[0,high],'--',color='#75818a',lw=1,label='Perfect forecast')
            ax.set(xlim=(-.2,high),ylim=(-.2,high),xlabel='Observed snowfall (inches)',ylabel='Forecast snowfall (inches)',aspect='equal')
            ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_title(f'{name}\n{windows}\n{len(group)} reports (samples may differ by model)',fontsize=10)
            save(fig,f'scatter_{number:03d}','scatter',name+' | '+windows.replace('\n','; '))
        diagnostics=percentile_diagnostics(d)
        if not diagnostics.empty:
            diagnostics.to_csv(out/'percentile_verification.csv',index=False)
            diag_keys=['model','init_time','forecast_start','forecast_end','obs_start','obs_end','ObType','window_basis']
            for number,(_,part) in enumerate(diagnostics.groupby(diag_keys,dropna=False,sort=True),start=1):
                part=part.sort_values('percentile');r=part.iloc[0]
                fig,ax=plt.subplots(figsize=(8,8))
                x=part.percentile.to_numpy();low=part.fraction_observed_below.to_numpy()*100
                high=part.fraction_observed_at_or_below.to_numpy()*100
                ax.plot([0,100],[0,100],'--',color='#89959c',label='Nominal percentile')
                ax.vlines(x,low,high,color=COLORS['nbm'],lw=3)
                ax.scatter(x,low,marker='v',color=COLORS['nbm'],label='Observed < forecast quantile')
                ax.scatter(x,high,marker='o',facecolors='white',edgecolors=COLORS['nbm'],label='Observed ≤ forecast quantile')
                for xx,yy,n in zip(x,high,part.n_reports):ax.annotate(f'n={n}',(xx,yy),xytext=(4,6),textcoords='offset points',fontsize=8)
                ax.set(xlim=(0,100),ylim=(-2,110),xlabel='Forecast percentile',ylabel='Percent of reports below quantile',
                       title=f"NBM percentile coverage | {r.ObType}\nInit {stamp(r.init_time)} · {r.window_basis}")
                ax.grid(alpha=.2)
                handles,legend_labels=ax.get_legend_handles_labels()
                fig.legend(handles,legend_labels,fontsize=8,loc='upper left',bbox_to_anchor=(.03,.205),borderaxespad=0)
                fig.text(.03,.015,f'Forecast: {stamp(r.forecast_start)} to {stamp(r.forecast_end)}. '
                         +observation_caption(r)+'.\n'
                         'Single-event, report-weighted diagnostic; repeated events are needed to assess calibration.\n'
                         'Bars show ties; observation types and accumulation windows are kept separate.',fontsize=8)
                fig.subplots_adjust(bottom=.30)
                save(fig,f'percentile_coverage_{number:03d}','percentile',f'NBM percentile coverage | {r.ObType} | init {stamp(r.init_time)}')
    # A portable browser index; all assets are local and no tile service is used.
    content=['<!doctype html><html><head><meta charset="utf-8"><title>Snowfall plots</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#253340}img{max-width:100%;height:auto}li{margin:8px 0}section{border-top:1px solid #ddd;margin-top:24px;padding-top:10px}small{color:#52606d}</style></head><body>',
             '<h1>'+html.escape(title)+'</h1>',
             '<p>All times UTC. Separate reports and model runs remain separate. Forecast windows may differ from observation windows. Nearby reports can share an archive point.</p>']
    content+=['<p>'+html.escape(w)+'</p>' for w in warnings]
    for kind,heading in [('overview','Station comparison charts'),('map','Forecast-error maps (NBM percentiles and NDFD)'),('scatter','Forecast versus observed'),('percentile','NBM percentile verification'),('station','Individual reports')]:
        content.append('<section><h2>'+heading+'</h2><ul>')
        for entry in [m for m in manifest if m['kind']==kind]:
            links=' · '.join('<a href="'+html.escape(f,quote=True)+'">'+Path(f).suffix[1:].upper()+'</a>' for f in entry['files'])
            content.append('<li>'+html.escape(entry['caption'])+' '+links+'</li>')
        content.append('</ul></section>')
    if not d.empty and quantile.any():content.append('<p><a href="percentile_verification.csv">Download percentile coverage and quantile loss table</a></p>')
    content.append('</body></html>')
    (out/'index.html').write_text('\n'.join(content),encoding='utf-8')
    settings=dict(comparisons=str(Path(comparisons).resolve()),individual=individual,rows_per_page=rows_per_page,
                  extent=extent,labels=labels,stations=stations,report_ids=report_ids,map_percentile=map_percentile,
                  map_percentiles=map_percentiles,
                  map_view=view_name,
                  error_colors=color_settings,
                  formats=list(formats),title=title,error_limit=error_limit)
    (out/'plot_manifest.json').write_text(json.dumps(dict(settings=settings,warnings=warnings,graphics=manifest),indent=2),encoding='utf-8')
    print(f'Created {len(manifest)} graphics. Open {out / "index.html"}.')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparisons',required=True)
    p.add_argument('--config',help='Verification JSON configuration; plotting settings are optional')
    p.add_argument('--map-view',help='Name from plotting.map_views; overrides plotting.map_view')
    p.add_argument('--output',default='snowfall_plots')
    p.add_argument('--station',action='append',help='Exact report name or mapped archive ID; repeat for more stations')
    p.add_argument('--report-id',action='append')
    p.add_argument('--no-individual',action='store_true')
    p.add_argument('--rows-per-page',type=int,default=10)
    p.add_argument('--map-extent',nargs=4,type=float,metavar=('WEST','EAST','SOUTH','NORTH'),help='Optional lon/lat bounds; WEST > EAST crosses the date line')
    p.add_argument('--labels',action='store_true',help='Label maps/scatterplots with report IDs')
    q=p.add_mutually_exclusive_group()
    q.add_argument('--map-percentile',type=float,help='Plot one percentile; default plots all available percentiles')
    q.add_argument('--map-percentiles',nargs='+',type=float,help='Plot a subset, e.g. 10 50 90')
    p.add_argument('--error-limit',type=float,help='Symmetric map color limit in inches; outliers saturate beyond the limits')
    p.add_argument('--map-cmap',help='Matplotlib colormap name, e.g. RdBu_r, BrBG, PuOr_r')
    p.add_argument('--error-range',nargs=2,type=float,metavar=('MIN','MAX'),help='Color range spanning zero; default -10 10')
    p.add_argument('--error-step',type=float,help='Color bin size in inches; default 2')
    p.add_argument('--error-levels',nargs='+',type=float,help='Explicit increasing bin boundaries spanning zero')
    p.add_argument('--formats',nargs='+',choices=['png','pdf'],default=['png'])
    p.add_argument('--title',default='Snowfall event verification')
    a=p.parse_args()
    config=json.loads(Path(a.config).read_text(encoding='utf-8')) if a.config else None
    plot_results(a.comparisons,a.output,individual=not a.no_individual,rows_per_page=a.rows_per_page,
                 extent=a.map_extent,labels=a.labels,stations=a.station,report_ids=a.report_id,
                 map_percentile=a.map_percentile,map_percentiles=a.map_percentiles,error_limit=a.error_limit,formats=a.formats,title=a.title,
                 config=config,map_view=a.map_view,map_cmap=a.map_cmap,error_range=a.error_range,error_step=a.error_step,error_levels=a.error_levels)


if __name__=='__main__':main()
