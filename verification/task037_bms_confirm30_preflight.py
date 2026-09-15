#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re
from pathlib import Path
import numpy as np, pandas as pd
SPANS=(1,2,4,8,16); CANON=(0,4,8,12); T=32
EXCLUDED={"HighJump","Mixing","Rafting"}
def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def parse_rows(val_list, frame_root):
 rows=[]
 with open(val_list,encoding='utf-8') as f:
  for idx,line in enumerate(f):
   z=line.split()
   if len(z)<3: continue
   vid=z[0]; dur=int(z[1]); lab=int(z[2]); parts=vid.split('_'); cls=parts[1]
   d=Path(frame_root)/cls/vid
   valid=d.is_dir() and dur>=2 and (d/f'image_{min(32,dur):05d}.jpg').is_file()
   if valid:
    rows.append(dict(video_path=vid,class_name=cls,duration=dur,label=lab,validation_list_index=idx,frame_dir=str(d)))
 return rows
def main(a):
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 pilot=Path(a.pilot_output); freeze=json.load(open(pilot/'task_bms_temporal_manifest_freeze.json'))
 dm=pilot/'task_bms_temporal_domain_manifest.csv'; um=pilot/'task_bms_temporal_unit_manifest.csv'
 checks=[]
 for name,p in [('pilot_domain_manifest',dm),('pilot_unit_manifest',um)]:
  got=sha256(p); exp=freeze['domain_manifest_sha256'] if name.startswith('pilot_domain') else freeze['unit_manifest_sha256']
  if got!=exp: raise SystemExit(f'{name} hash mismatch {got} != {exp}')
  checks.append(dict(check=name,expected=exp,observed=got,status='PASS'))
 domains=pd.read_csv(dm); units=pd.read_csv(um)
 if len(units)!=36 or len(domains[domains.selected])!=9: raise SystemExit('pilot identity mismatch')
 rows=parse_rows(a.val_list,a.frame_root)
 by={}
 for r in rows: by.setdefault(r['class_name'],[]).append(r)
 eligible=sorted(c for c,v in by.items() if len(v)>=3 and c not in EXCLUDED)
 rng=np.random.RandomState(3407); chosen_idx=sorted(rng.choice(len(eligible),10,replace=False).tolist()); chosen=[eligible[i] for i in chosen_idx]
 selected=[]
 for ci,c in zip(chosen_idx,chosen):
  cand=sorted(by[c],key=lambda r:r['video_path'])
  rr=np.random.RandomState(3407+ci); ix=sorted(rr.choice(len(cand),3,replace=False).tolist())
  for pos,j in enumerate(ix):
   r=dict(cand[j]); r.update(class_index=int(ci),within_class_position=int(pos),selection_seed=int(3407+ci),discovery_class_excluded=False)
   selected.append(r)
 # arrange manifest parity so each GPU shard contains 5 complete classes
 selected.sort(key=lambda r:(chosen.index(r['class_name']),r['video_path']))
 for r in selected:
  ci=chosen.index(r['class_name']); pos=int(r['within_class_position'])
  if ci<5: mo=2*(ci*3+pos)
  else: mo=2*((ci-5)*3+pos)+1
  r['manifest_order']=mo
 selected.sort(key=lambda r:r['manifest_order'])
 vm=pd.DataFrame(selected)[['class_name','class_index','video_path','validation_list_index','within_class_position','selection_seed','discovery_class_excluded','manifest_order','duration','label','frame_dir']]
 if len(vm)!=30 or vm.class_name.nunique()!=10 or vm.groupby('class_name').size().min()!=3: raise SystemExit('manifest cardinality mismatch')
 vm.to_csv(out/'task_bms_confirm30_video_manifest.csv',index=False)
 # exact loader list rows, preserving original labels
 for parity in (0,1):
  shard=vm[vm.manifest_order%2==parity].sort_values('manifest_order')
  with open(out/f'task_bms_confirm30_gpu{parity}_val_list.txt','w') as f:
   for _,r in shard.iterrows(): f.write(f"{r.video_path} {int(r.duration)} {int(r.label)}\n")
 cm=[]
 for s in SPANS:
  for p in CANON:
   q,r=divmod(p,s); left=2*q*s+r; right=left+s
   cm.append(dict(span=s,canonical_pair_index=p,frame_a=left,frame_b=right))
 ctx=pd.DataFrame(cm); ctx.to_csv(out/'task_bms_confirm30_context_manifest.csv',index=False)
 # compare authoritative pilot enumeration
 pjson=json.load(open(pilot/'task_bms_temporal_intervention_manifest.json'))
 ref={(int(x.get('block_size',x.get('span'))),int(x['pair_index'])):(int(x.get('frame_left',x.get('left_start'))),int(x.get('frame_right',x.get('right_start')))) for x in pjson['interventions']}
 for _,r in ctx.iterrows():
  if ref.get((int(r.span),int(r.canonical_pair_index)))!=(int(r.frame_a),int(r.frame_b)): raise SystemExit('context identity mismatch')
 checks += [dict(check='30_video_manifest_deterministic',expected='30',observed=str(len(vm)),status='PASS'),dict(check='discovery_class_exclusion',expected=sorted(EXCLUDED),observed=sorted(set(vm.class_name)&EXCLUDED),status='PASS'),dict(check='10_class_identity',expected=10,observed=int(vm.class_name.nunique()),status='PASS'),dict(check='3_videos_per_class',expected='3 each',observed=vm.groupby('class_name').size().to_dict(),status='PASS'),dict(check='20_context_identity',expected=20,observed=len(ctx),status='PASS'),dict(check='canonical_indices',expected=list(CANON),observed=sorted(ctx.canonical_pair_index.unique().tolist()),status='PASS'),dict(check='5_span_identity',expected=list(SPANS),observed=sorted(ctx.span.unique().tolist()),status='PASS'),dict(check='pilot_frozen_domains_units',expected='9/36',observed=f'{int(domains.selected.sum())}/{len(units)}',status='PASS')]
 pd.DataFrame(checks).to_csv(out/'task_bms_confirm30_identity_audit.csv',index=False)
 manifest={'task':'TASK037_CONFIRM30','seed':3407,'excluded_discovery_classes':sorted(EXCLUDED),'selected_classes':chosen,'class_indices':chosen_idx,'num_videos':30,'videos_per_class':3,'manifest_sha256':sha256(out/'task_bms_confirm30_video_manifest.csv'),'context_manifest_sha256':sha256(out/'task_bms_confirm30_context_manifest.csv'),'pilot_domain_manifest_sha256':sha256(dm),'pilot_unit_manifest_sha256':sha256(um),'selection_algorithm':'eligible classes sorted lexicographically; RandomState(3407) selects 10; per-class RandomState(3407+class_index) selects 3 sorted paths; manifest parity assigns first five classes to even and last five to odd orders for complete 15-video GPU shards','contexts':cm}
 json.dump(manifest,open(out/'task_bms_confirm30_manifest.json','w'),indent=2)
 print(json.dumps(manifest,indent=2))
 print(vm[['manifest_order','class_name','video_path']].to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--pilot-output',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--val-list',required=True); p.add_argument('--frame-root',required=True); main(p.parse_args())
