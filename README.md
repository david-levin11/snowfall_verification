# Standalone snowfall verification

This first version compares QC station reports with station-sampled Parquet archives. It does not download forecasts or change the archiver. Requires Python 3.9+, pandas, numpy and pyarrow:

```bash
python -m pip install pandas numpy pyarrow
```

## Your AFC event

November 15 00 UTC through November 17 18 UTC, 2026 is **66 hours**. An NDFD forecast can cover this with eleven contiguous six-hour increments. NBM must use a native product: a 72-hour window ending November 18 00 UTC adds six hours after the event; one ending November 17 18 UTC adds six hours before it. Either may be useful if those extra hours were dry. The script reports these differences; it cannot determine whether the mismatch was meteorologically harmless.

Do not divide a 72-hour snowfall forecast by 72 and multiply by 66. This script neither prorates totals nor sums NBM percentiles. It chooses windows from timing alone, never by which forecast best matches the observed amount.

## Observation windows first

The supplied CSV has 81 reports: 12 LSR, 2 COOP, 9 CoCoRaHS and 58 SNOWDEPTH_SITE. The datetime column is a report timestamp, not a known accumulation endpoint. Most reports have no datetime at all. The extraction dates do not prove that every total covers those dates.

The inspected GetSnowfallData.py retains LSR valid time without duration, sums COOP reports, aggregates CoCoRaHS by date, and derives snow-depth changes for snow-depth sites. This combined CSV cannot recover the original accumulation windows. The snow-depth method also smooths/interpolates or uses observations at or before requested boundaries, and clips negative changes to zero. A zero can therefore represent a nonpositive depth change, rather than verified absence of snowfall. Settling, melting and redistribution can affect depth changes. QC of values does not restore missing time metadata.

Prepare an editable copy:

```bash
python verify_snowfall.py --obs SnowfallObsTEST.csv --prepare snowfall_review.csv
```

Fill `obs_start` and `obs_end` with UTC timestamps, for example `2026-11-15T00:00:00Z`. Use actual accumulation boundaries where known. Keep both blank for unknown windows: these reports are listed as unmatched. Set `include=no` to exclude a report, and use `notes` to document QC decisions. Preserve `report_id` when reordering rows. Reports are never automatically combined or deduplicated.

For an exploratory event comparison, `--assume-event-window` fills pairs of missing boundaries with the event boundaries you supply. This is an explicit scientific assumption, recorded as `window_basis=assumed_event`. Per-report windows override it. A partially supplied window is an error. Do not use this assumption indiscriminately for early LSRs: the event might have continued after the report.

## Spatial matching

Supply the station metadata used to create the archive: `stid` (or `station_id`), `latitude`, `longitude`. The script matches report `stid` where available, otherwise `stationname`, to archive station IDs, case insensitively. For other reports, fill `archive_station_id` in the review CSV. Metadata coordinates and distance are retained for audit. An explicit mapping is authoritative even at a large distance; review its distance yourself.

Optionally use `--nearest-km 5` for unmatched report IDs. This searches metadata locations within 5 km. The nearest metadata station may lack snowfall forecasts and then remains unmatched; the script does not keep moving outward to find data. Distances refer to the archived station location, not the actual sampled grid cell. In Alaska terrain, proximity alone does not establish representativeness; use manual mappings/exclusions as appropriate. No automatic elevation matching is performed. Multiple reports may map to one archive point, which can overweight that location.

## Configure forecasts

Edit example_config.json. Remove products/columns you do not have and set actual archive paths. Relative file paths resolve relative to the configuration file. `files` accepts a path, glob, or list of globs. List adjacent months when a window crosses a file boundary. Only point the glob at snowfall files for that specific product. The script reads the selected files into memory; restrict files to relevant months.

Each source explicitly declares model, accumulation hours, units and value columns. NBM archive columns inspected in alaska_verification/main are `snow_p10`, `snow_p50`, etc.; NDFD is `snow6hr`. Archive amounts are already inches. `hours` describes the actual product, not a desired aggregation length. Accepted units: in, cm, mm, m. Observed units default to inches (`--obs-units` can change them).

`valid_time` must be the accumulation END; start is end minus product duration. This follows the current archiver's endpoint convention. Historical archives with incorrect timestamps need repair before use; this script does not guess timestamp corrections. Inputs require station_id, init_time, valid_time and configured value columns. Identical duplicate rows collapse; conflicting duplicate keys fail explicitly. Missing or negative observed amounts are excluded; negative/infinite forecast values fail. Missing NDFD values or gaps break a chain.

For your event, after reviewing which reports represent event totals:

```bash
python verify_snowfall.py \
  --obs snowfall_review.csv \
  --stations alaska_all_active_synoptic_station_metadata.csv \
  --config example_config.json \
  --event-start 2026-11-15T00:00Z \
  --event-end 2026-11-17T18:00Z \
  --assume-event-window \
  --endpoint-tolerance-h 6 \
  --duration-tolerance-h 6 \
  --init 2026-11-14T00:00Z \
  --output afc_snowfall_results
```

Use an issuance actually present in your files. Omit `--init` to compare each available issuance separately. Issuances after the observation window starts are rejected to prevent hindsight comparisons. Every NDFD chain stays within one station and one init_time; cycles are never stitched together. NDFD chains may have any positive multiple of six hours. NBM uses only individual native products.

Endpoint tolerance independently bounds the absolute start and end differences; duration tolerance additionally bounds the length difference. Defaults require exact endpoints. At 6h tolerance, a 66h report could have 60h or 72h candidates; configure only the products you want considered. Start/end offsets are forecast minus observation (positive means later). Windows must overlap.

## Output

- comparisons.csv: one best timing match per report/model/issuance/field, with observed and forecast amounts, signed error (forecast minus observed), absolute error, duration, boundary offsets, overlap/union fraction, lead time, mapping distance, observation type and assumptions.
- candidates.csv: every eligible timing candidate, its rank, and selected flag. Ties resolve by duration difference then earliest start/end. Review this for alternative 72h alignments. It contains eligible candidates only, not every rejected forecast window.
- unmatched.csv: excluded reports and unmatched report/forecast-series combinations with reasons. A report with no station forecasts gets one rejection; an absent model/issuance series cannot be enumerated as a rejection.
- summary.csv: report count, distinct archive station count, mean error and MAE, separately by model, issuance, field, timing class, window basis and observation type. These are descriptive report-weighted event statistics. Percentile errors describe percentile values; they are not a full probabilistic skill assessment. NBM and NDFD may select different windows/report subsets, so their summaries are not necessarily a paired model comparison.
- run_settings.json: inputs and matching settings.

Depth-change results stay separate through ObType. Exact and approximate matches also stay separate. Do not treat nearby reports or multiple percentiles as independent verification cases.

## Validation and remaining production check

Included tests cover complete/gapped NDFD chains, cycle isolation, native NBM windows, tolerance rejection, amount-independent selection, issuance cutoff, missing report windows, duplicate conflicts, spatial limits and a Parquet CLI round trip. No real November snowfall Parquet file was supplied, so actual archive/event results remain untested. First run with one issuance and inspect comparisons/unmatched records before expanding.

For future GetSnowfallData.py exports, preserve station IDs, measurement kind, actual accumulation start/end, report time, source IDs, missing-data coverage and QC notes before aggregation. Those fields would improve this workflow much more than increasing matching tolerances.
