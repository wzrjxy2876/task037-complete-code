from pathlib import Path
import json, hashlib, pandas as pd, numpy as np
OUT=Path('/data/jixinye25/work1/output/task037_bms_confirm30')
EXPECTED='4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63'
def test_manifest_identity():
 d=pd.read_csv(OUT/'task_bms_confirm30_video_manifest.csv'); assert len(d)==30; assert d.class_name.nunique()==10; assert d.groupby('class_name').size().eq(3).all(); assert not d.class_name.isin(['HighJump','Mixing','Rafting']).any(); assert sorted(d.manifest_order.tolist())==list(range(30))
def test_context_identity():
 d=pd.read_csv(OUT/'task_bms_confirm30_context_manifest.csv'); assert len(d)==20; assert sorted(d.span.unique())==[1,2,4,8,16]; assert set(d.canonical_pair_index)=={0,4,8,12}; assert all(d.frame_b-d.frame_a==d.span)
def test_checkpoint_identity():
 x=json.load(open(OUT/'task_bms_confirm30_checkpoint_identity.json')); assert x['checkpoint_sha256']==EXPECTED; assert x['identity_status']=='PASS'; assert x['dtype']=='torch.float32'; assert x['amp'] is False
def test_gpu_shards_and_merged_identity():
 ds=[]
 for p in (0,1):
  x=pd.read_csv(OUT/f'shards/gpu{p}/task040_raw_records.csv'); assert len(x)==10800; assert x.video_id.nunique()==15; assert x.unit_global_index.nunique()==36; assert len(x[['block_size','pair_index']].drop_duplicates())==20; ds.append(set(x.video_id))
 assert ds[0].isdisjoint(ds[1]) and len(ds[0]|ds[1])==30
 x=pd.read_csv(OUT/'task_bms_confirm30_conditioned_damage.csv'); assert len(x)==21600; assert x.video_key.nunique()==30 and x.global_index.nunique()==36
def test_signed_arithmetic_and_decision():
 x=pd.read_csv(OUT/'task_bms_confirm30_reversal_pairs.csv'); assert ((x.D_ori_i-x.D_ori_j)*(x.D_rel_i-x.D_rel_j)<0).eq(x.strict_reversal).all(); s=json.load(open(OUT/'task_bms_confirm30_summary.json')); assert s['decision'] in {'INDEPENDENT_BMS_TEMPORAL_RELATION_MOTIVATION_CONFIRMED','INDEPENDENT_BMS_TEMPORAL_RELATION_MOTIVATION_PARTIALLY_CONFIRMED','INDEPENDENT_BMS_TEMPORAL_RELATION_MOTIVATION_NOT_CONFIRMED'}
def test_severity_same_span_bootstrap():
 s=pd.read_csv(OUT/'task_bms_confirm30_severity_matching.csv'); assert len(s)>0; q=pd.read_csv(OUT/'task_bms_confirm30_same_span_results.csv'); assert len(q)==30*9*5*6; b=pd.read_csv(OUT/'task_bms_confirm30_bootstrap.csv'); assert len(b)==10000; assert b.replicate.tolist()==list(range(10000))
