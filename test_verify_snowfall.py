import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import verify_snowfall as v

T = v.utc('2026-11-15')

def forecasts(model='ndfd', periods=11):
    return pd.DataFrame([dict(model=model,station_id='TEST',init_time=T-pd.Timedelta('24h'),start=T+pd.Timedelta(hours=i*6),end=T+pd.Timedelta(hours=(i+1)*6),forecast_in=1.,field='snow6hr',period_h=6) for i in range(periods)])

def args(**kw):
    return SimpleNamespace(**dict(dict(endpoint_tolerance_h=0, duration_tolerance_h=None,nearest_km=None), **kw))

def obs():
    return pd.DataFrame([dict(report_id='R1', stationname='TEST',Lat=60,Lon=-150,snowfall=11, observed_in=11,include='yes',archive_station_id='',obs_start=T,obs_end=T+pd.Timedelta('66h'),window_basis='explicit',ObType='COOP')])

META = pd.DataFrame([dict(station_id='TEST',latitude=60,longitude=-150)]).set_index('station_id')

class TestSnow(unittest.TestCase):
    def test_complete_ndfd(self):
        c,u=v.compare(obs(),forecasts(),META,args())
        self.assertEqual(c.iloc[0].forecast_in,11)
        self.assertEqual(c.iloc[0].n_increments,11)
        self.assertTrue(u.empty)
    def test_gap(self):
        c,u=v.compare(obs(),forecasts().drop(index=3),META,args())
        self.assertTrue(c.empty)
        self.assertEqual(u.iloc[0].reason,'no_complete_window_within_tolerances')
    def test_cycles_not_stitched(self):
        f=forecasts();f.loc[6:,'init_time']=T-pd.Timedelta('12h')
        c,u=v.compare(obs(),f,META,args());self.assertTrue(c.empty)
    def test_nbm_native(self):
        f=forecasts('nbm',12)
        self.assertEqual(len(list(v.windows(f,'nbm',T,T+pd.Timedelta('66h'),0))),12)
        c,u=v.compare(obs(),f,META,args());self.assertTrue(c.empty)
    def test_nbm_timing_and_selection(self):
        f=forecasts('nbm',2);f['period_h']=72
        f['start']=[T-pd.Timedelta('6h'),T];f['end']=f.start+pd.Timedelta('72h')
        f['forecast_in']=[100.,11.]
        c,u=v.compare(obs(),f,META,args(endpoint_tolerance_h=6,duration_tolerance_h=6))
        self.assertEqual(c[c.selected].iloc[0].forecast_in,100.)
        self.assertEqual(len(c),2)
        c,u=v.compare(obs(),f,META,args(endpoint_tolerance_h=6,duration_tolerance_h=5));self.assertTrue(c.empty)
    def test_hindsight(self):
        f=forecasts();f['init_time']=T+pd.Timedelta('1h')
        c,u=v.compare(obs(),f,META,args());self.assertEqual(u.iloc[0].reason,'issuance_after_observation_start')
    def test_unknown_window(self):
        o=obs();o['obs_start']='';o['obs_end']=''
        c,u=v.compare(o,forecasts(),META,args());self.assertEqual(u.iloc[0].reason,'missing_observation_window')
    def test_spatial(self):
        r=obs().iloc[0].copy();r.stationname='LSR';r.Lat=61
        self.assertEqual(v.map_station(r,META,None)[3],'no_station_id_match')
        self.assertEqual(v.map_station(r,META,5)[3],'no_station_within_distance')
        self.assertEqual(v.map_station(r,META,120)[0],'TEST')
    def test_parquet_cli_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=forecasts().rename(columns={'end':'valid_time','forecast_in':'snow6hr'})
            f.to_parquet(p/'f.parquet',index=False)
            conf={'sources':[dict(model='ndfd',hours=6,files='f.parquet',columns=['snow6hr'],units='in')]}
            (p/'config.json').write_text(json.dumps(conf))
            obs().to_csv(p/'obs.csv',index=False);META.to_csv(p/'meta.csv')
            subprocess.run([sys.executable,str(Path(v.__file__)), '--obs',str(p/'obs.csv'),'--stations',str(p/'meta.csv'),'--config',str(p/'config.json'),'--output',str(p/'out')],check=True,capture_output=True)
            self.assertEqual(pd.read_csv(p/'out/comparisons.csv').iloc[0].forecast_in,11)
            bad=f.iloc[[0]].copy();bad.snow6hr=3
            pd.concat([f,bad]).to_parquet(p/'f.parquet',index=False)
            with self.assertRaisesRegex(ValueError,'Conflicting duplicate'):
                v.load_sources(conf,p,SimpleNamespace(init=None))

if __name__=='__main__': unittest.main()
