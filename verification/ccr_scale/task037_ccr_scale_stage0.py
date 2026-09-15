#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
import pandas as pd

_HERE=Path(__file__).resolve()
# In the repository this file lives under verification/ccr_scale; during
# local preparation it lives under qing/work. Resolve both layouts without
# relying on a machine-specific absolute path.
ROOT=_HERE.parents[2] if _HERE.parent.name=='ccr_scale' else _HERE.parents[1]
OUT=ROOT/'outputs/task037_ccr_scale'; SRC=ROOT/'outputs/task037_ccr_scale_source'; CONF=ROOT/'outputs/task037_bms_confirm30'
OLD_DOM={350,376,401,271,287,296,32,79,108}; SPANS=[1,2,4,8,16]; PAIRS=[0,8]

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def category(r):
 a=int(r.attention_count); f=int(r.ffn_count)
 return 'MIXED' if a>0 and f>0 else ('AA' if a>0 else ('FF' if f>0 else 'NONE'))

def select_mixed(g):
 aa=g[g.unit_type=='attention_head'].sort_values('global_index'); ff=g[g.unit_type=='ffn_neuron'].sort_values('global_index'); target=min(4,len(g)); choices=[]
 for na in range(target+1):
  nf=target-na
  if na<=len(aa) and nf<=len(ff):
   inds=sorted(list(aa.head(na).global_index)+list(ff.head(nf).global_index)); choices.append(((0 if (na==2 and nf==2) else 1,abs(na-nf),inds),na,nf))
 _,na,nf=min(choices,key=lambda z:z[0]); return pd.concat([aa.head(na),ff.head(nf)]).sort_values('global_index')

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 scores=pd.read_csv(SRC/'initial_candidate_scores.csv')
 if scores.global_index.duplicated().any(): raise AssertionError('duplicate global_index')
 unit=scores[['global_index','domain_id','layer','unit_type','unit_index']].copy()
 for c in ['global_index','domain_id','unit_index']: unit[c]=unit[c].astype(int)
 g=unit.groupby('domain_id').agg(full_valid_domain_size=('global_index','size'),attention_count=('unit_type',lambda s:int((s=='attention_head').sum())),ffn_count=('unit_type',lambda s:int((s=='ffn_neuron').sum()))).reset_index()
 g['category']=g.apply(category,axis=1); g['eligible']=((g.category=='AA')&(g.attention_count>=3))|((g.category=='FF')&(g.ffn_count>=3))|((g.category=='MIXED')&(g.attention_count>=1)&(g.ffn_count>=1)&(g.full_valid_domain_size>=3)); g['old_ccr_domain']=g.domain_id.isin(OLD_DOM)
 selected=[]
 for cat in ['AA','FF','MIXED']:
  elig=g[(g.category==cat)&g.eligible&~g.old_ccr_domain].sort_values(['full_valid_domain_size','domain_id'],ascending=[False,True]); selected += [(cat,int(x)) for x in elig.head(10).domain_id]
  if len(elig)<10: print('LIMITATION',cat,len(elig))
 if len(selected)<20: raise AssertionError(f'selected domains {len(selected)}')
 selset={d for _,d in selected}; order={(cat,d):i for i,(cat,d) in enumerate(selected)}
 dg=g[g.domain_id.isin(selset)].copy(); dg['selection_category']=dg.category; dg['selection_order']=[order[(r.category,int(r.domain_id))] for r in dg.itertuples()]; dg['old_ccr_domain']=False; dg['unit_count_manifest']=0; dg=dg.sort_values('selection_order')
 rows=[]
 for cat,did in selected:
  gg=unit[unit.domain_id==did].sort_values('global_index')
  chosen=gg[gg.unit_type=='attention_head'].head(4) if cat=='AA' else (gg[gg.unit_type=='ffn_neuron'].head(4) if cat=='FF' else select_mixed(gg))
  if len(chosen)<3: raise AssertionError(f'domain {did} chosen {len(chosen)}')
  rule='AA_first4_global_index' if cat=='AA' else ('FF_first4_global_index' if cat=='FF' else 'MIXED_balanced_2plus2_or_closest')
  dg.loc[dg.domain_id==did,'unit_count_manifest']=len(chosen)
  for rank,r in enumerate(chosen.itertuples(index=False)): rows.append({'category':cat,'domain_id':did,'global_index':int(r.global_index),'layer':str(r.layer),'unit_type':str(r.unit_type),'unit_index':int(r.unit_index),'selection_rank':rank,'unit_selection_rule':rule})
 dg.to_csv(OUT/'task_ccr_scale_domain_manifest.csv',index=False); pd.DataFrame(rows).sort_values(['category','domain_id','global_index']).to_csv(OUT/'task_ccr_scale_unit_manifest.csv',index=False)
 vm=pd.read_csv(CONF/'task_bms_confirm30_video_manifest.csv');
 if len(vm)!=30 or vm.manifest_order.tolist()!=list(range(30)): raise AssertionError('video identity')
 vm['gpu_shard']=vm.manifest_order%2; vm['gpu_shard_label']=vm.gpu_shard.map({0:'gpu0_even',1:'gpu1_odd'}); classes=sorted(vm.class_index.unique()); pos={c:i for i,c in enumerate(classes)}
 vm['n9a']=vm.apply(lambda r: pos[int(r.class_index)]<9 and int(r.within_class_position)==pos[int(r.class_index)]%3,axis=1)
 vm['n9b']=vm.apply(lambda r: 1<=pos[int(r.class_index)]<=9 and int(r.within_class_position)==(pos[int(r.class_index)]%3+1)%3,axis=1)
 vm['n9c']=vm.apply(lambda r: pos[int(r.class_index)]<9 and int(r.within_class_position)==(pos[int(r.class_index)]%3+2)%3,axis=1)
 if vm[['n9a','n9b','n9c']].sum().tolist()!=[9,9,9]: raise AssertionError('N9 size')
 vm['source_manifest_sha256']=sha(CONF/'task_bms_confirm30_video_manifest.csv'); vm.to_csv(OUT/'task_ccr_scale_video_manifest.csv',index=False)
 oldc=pd.read_csv(CONF/'task_bms_confirm30_context_manifest.csv'); crow=[{'context_id':0,'context_kind':'original','span':0,'canonical_pair_index':-1,'pair_index':-1,'frame_a':-1,'frame_b':-1,'frame_pair':'original'}]
 for s in SPANS:
  for pidx in PAIRS:
   rr=oldc[(oldc.span==s)&(oldc.canonical_pair_index==pidx)]
   if len(rr)!=1: raise AssertionError(f'missing context {(s,pidx)}')
   r=rr.iloc[0]; crow.append({'context_id':len(crow),'context_kind':'relation','span':int(s),'canonical_pair_index':int(pidx),'pair_index':int(pidx),'frame_a':int(r.frame_a),'frame_b':int(r.frame_b),'frame_pair':f'{int(r.frame_a)}-{int(r.frame_b)}'})
 cm=pd.DataFrame(crow); cm.to_csv(OUT/'task_ccr_scale_context_manifest.csv',index=False)
 counts={cat:sum(c==cat for c,d in selected) for cat in ['AA','FF','MIXED']}
 audits=[{'check':'old_9_exclusion','status':'PASS' if not selset&OLD_DOM else 'FAIL','observed':sorted(selset&OLD_DOM),'expected':[]},{'check':'domain_counts_by_category','status':'PASS' if counts['AA']==10 and counts['FF']==10 and counts['MIXED']<=10 else 'FAIL','observed':counts,'expected':{'AA':10,'FF':10,'MIXED':'up to 10; all eligible'}},{'check':'unit_manifest_count','status':'PASS' if 90<=len(rows)<=120 else 'FAIL','observed':len(rows),'expected':'90-120'},{'check':'video_manifest_30','status':'PASS' if len(vm)==30 else 'FAIL','observed':len(vm),'expected':30},{'check':'gpu_even_odd_15_each','status':'PASS' if vm.gpu_shard.value_counts().to_dict()=={0:15,1:15} else 'FAIL','observed':vm.gpu_shard.value_counts().to_dict(),'expected':{0:15,1:15}},{'check':'contexts_10_relation_plus_original','status':'PASS' if len(cm)==11 else 'FAIL','observed':len(cm),'expected':11},{'check':'pair_identity_0_8','status':'PASS' if set(zip(cm[cm.context_kind=='relation'].span,cm[cm.context_kind=='relation'].canonical_pair_index))=={(s,p) for s in SPANS for p in PAIRS} else 'FAIL','observed':[(int(r.span),int(r.canonical_pair_index)) for r in cm[cm.context_kind=='relation'].itertuples()],'expected':[(s,p) for s in SPANS for p in PAIRS]},{'check':'n9a_b_c_sizes','status':'PASS' if vm[['n9a','n9b','n9c']].sum().tolist()==[9,9,9] else 'FAIL','observed':vm[['n9a','n9b','n9c']].sum().tolist(),'expected':[9,9,9]}]
 pd.DataFrame(audits).to_csv(OUT/'task_ccr_scale_identity_audit.csv',index=False)
 meta={'task':'TASK037_CCR_SCALE','source_scores_sha256':sha(SRC/'initial_candidate_scores.csv'),'source_video_manifest_sha256':sha(CONF/'task_bms_confirm30_video_manifest.csv'),'source_context_manifest_sha256':sha(CONF/'task_bms_confirm30_context_manifest.csv'),'old_ccr_domains':sorted(OLD_DOM),'selected_domains':selected,'selected_unit_count':len(rows),'video_count':len(vm),'context_count':len(cm),'n9_counts':{k:int(vm[k].sum()) for k in ['n9a','n9b','n9c']}}
 (OUT/'task_ccr_scale_source_identity.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
