import argparse, hashlib, itertools, json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

def sha256(p):
 h=hashlib.sha256(); f=open(p,'rb')
 for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 f.close(); return h.hexdigest()
def metrics_from_values(x,y):
 x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
 if len(x)<2: return dict(spearman=np.nan,kendall=np.nan,pairwise_ordering_accuracy=np.nan,comparable_pairs=0,correct_pairs=0,top1_identity=False,bottom1_identity=False)
 rx=rankdata(x,method='average'); ry=rankdata(y,method='average'); sp=float(np.corrcoef(rx,ry)[0,1]) if len(x)>1 and np.std(rx)>0 and np.std(ry)>0 else np.nan
 comp=cor=0
 for i,j in itertools.combinations(range(len(x)),2):
  a=x[i]-x[j]; b=y[i]-y[j]
  if a==0 or b==0: continue
  comp+=1; cor+=int(np.sign(a)==np.sign(b))
 return dict(spearman=sp,kendall=(2*cor-comp)/comp if comp else np.nan,pairwise_ordering_accuracy=cor/comp if comp else np.nan,comparable_pairs=comp,correct_pairs=cor,top1_identity=int(np.argmax(x)==np.argmax(y)),bottom1_identity=int(np.argmin(x)==np.argmin(y)))
def metric_pairs(g):
 z=g[g.strict_comparable.astype(bool)]; comp=len(z); cor=int(z.agreement.astype(bool).sum()); return dict(spearman=np.nan,kendall=(2*cor-comp)/comp if comp else np.nan,pairwise_ordering_accuracy=cor/comp if comp else np.nan,comparable_pairs=comp,correct_pairs=cor)
def add_pair_groups(p,groups,label):
 rows=[]
 for key,g in p.groupby(groups,sort=True,dropna=False):
  if not isinstance(key,tuple): key=(key,)
  r={'analysis':label}; r.update(dict(zip(groups,key))); r.update(metric_pairs(g)); rows.append(r)
 return pd.DataFrame(rows)
def rev_counts(g):
 o=g.oracle_reversal.astype(bool); q=g.proxy_reversal.astype(bool); tp=int((o&q).sum()); fp=int((~o&q).sum()); tn=int((~o&~q).sum()); fn=int((o&~q).sum()); p=tp/(tp+fp) if tp+fp else np.nan; r=tp/(tp+fn) if tp+fn else np.nan; f=2*p*r/(p+r) if np.isfinite(p) and np.isfinite(r) and p+r else np.nan; return dict(TP=tp,FP=fp,TN=tn,FN=fn,precision=p,recall=r,F1=f,balanced_accuracy=0.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else np.nan,MCC=(tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) else np.nan,oracle_reversals=tp+fn,proxy_reversals=tp+fp)
def rev_groups(r,groups,label):
 rows=[]
 for key,g in r.groupby(groups,dropna=False,sort=True):
  if not isinstance(key,tuple): key=(key,)
  d={'analysis':label}; d.update(dict(zip(groups,key))); d.update(rev_counts(g)); rows.append(d)
 return pd.DataFrame(rows)
def main(args):
 out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); raw=pd.read_csv(Path(args.shard0)/'task_context_proxy_raw.csv'); raw=pd.concat([raw,pd.read_csv(Path(args.shard1)/'task_context_proxy_raw.csv')],ignore_index=True); raw['class_name']=raw['class'].astype(str); um=pd.read_csv(args.unit_manifest); raw=raw.merge(um[['global_index','category']].drop_duplicates(),on='global_index',how='left',validate='many_to_one'); rel=pd.read_csv(Path(args.confirmation_dir)/'task_bms_confirm30_conditioned_damage.csv'); delta={(str(r.video_key),int(r.span),int(r.pair_index)):float(r.Delta_model) for r in rel.drop_duplicates(['video_key','span','pair_index']).itertuples(index=False)}; raw['Delta_model']=[0.0 if r.context_id==0 else delta[(str(r.video_key),int(r.span),int(r.pair_index))] for r in raw.itertuples(index=False)]
 if len(raw)!=22680 or raw.video_key.nunique()!=30 or raw.global_index.nunique()!=36: raise AssertionError('merged identity')
 raw.to_csv(out/'task_context_proxy_raw.csv',index=False); vm=pd.read_csv(args.video_manifest); vm['source_manifest_sha256']=sha256(args.video_manifest); vm.to_csv(out/'task_context_proxy_manifest.csv',index=False)
 # conditional pairwise rows already computed by GPU-independent deterministic arithmetic
 pr=[]
 for (v,d,cid),g in raw[raw.context_id>0].groupby(['video_key','domain_id','context_id'],sort=True):
  ids=sorted(g.global_index.astype(int)); q=g.iloc[0]; vals={int(r.global_index):r for r in g.itertuples(index=False)}
  for i,j in itertools.combinations(ids,2):
   x=vals[i]; y=vals[j]; od=float(x.oracle_signed_damage)-float(y.oracle_signed_damage); pdif=float(x.proxy_signed_damage)-float(y.proxy_signed_damage); ti='A' if x.unit_type=='attention_head' else 'F'; tj='A' if y.unit_type=='attention_head' else 'F'; pr.append({'video_key':v,'manifest_order':int(q.manifest_order),'class':str(q.class_name),'class_name':str(q.class_name),'class_index':int(q.class_index),'domain_id':int(d),'category':str(q.category),'context_id':int(cid),'span':int(q.span),'pair_index':int(q.pair_index),'frame_pair':str(q.frame_pair),'global_index_i':i,'global_index_j':j,'pair_type':ti+tj,'type_i':x.unit_type,'type_j':y.unit_type,'stage_i':int(x.stage),'stage_j':int(y.stage),'oracle_diff':od,'proxy_diff':pdif,'oracle_sign':int(np.sign(od)),'proxy_sign':int(np.sign(pdif)),'strict_comparable':od!=0 and pdif!=0,'agreement':od!=0 and pdif!=0 and np.sign(od)==np.sign(pdif)})
 pair=pd.DataFrame(pr); pair.to_csv(out/'task_context_proxy_pairwise_order.csv',index=False)
 # original validation, only 270 groups
 ov=[]
 for (v,d),g in raw[raw.context_id==0].groupby(['video_key','domain_id'],sort=True):
  q=g.iloc[0]; m=metrics_from_values(g.proxy_signed_damage,g.oracle_signed_damage); ov.append(dict({'analysis':'original_domain','video_key':v,'domain_id':int(d),'category':str(q.category)},**m))
 origv=pd.DataFrame(ov); origv=pd.concat([origv,[] if len(origv)==0 else pd.DataFrame([dict({'analysis':'original_pooled_category','category':c},**metrics_from_values(raw[(raw.context_id==0)&(raw.category==c)].proxy_signed_damage,raw[(raw.context_id==0)&(raw.category==c)].oracle_signed_damage)) for c in sorted(raw.category.dropna().unique())])],ignore_index=True); origv.to_csv(out/'task_context_proxy_original_validation.csv',index=False)
 cv=add_pair_groups(pair,['video_key','domain_id','context_id','span','category'],'conditioned_domain'); cv=pd.concat([cv,add_pair_groups(pair,['category'],'conditioned_pooled_category')],ignore_index=True); cv.to_csv(out/'task_context_proxy_conditioned_validation.csv',index=False)
 # fixed and perfect-static pair tables
 base=raw[raw.context_id==0][['video_key','domain_id','global_index','proxy_signed_damage','oracle_signed_damage']].rename(columns={'proxy_signed_damage':'fixed_i','oracle_signed_damage':'perfect_i'}); f=pair.merge(base.rename(columns={'global_index':'global_index_i'}),on=['video_key','domain_id','global_index_i']).merge(base.rename(columns={'global_index':'global_index_j','fixed_i':'fixed_j','perfect_i':'perfect_j'}),on=['video_key','domain_id','global_index_j']); f['fixed_diff']=f.fixed_i-f.fixed_j; f['perfect_diff']=f.perfect_i-f.perfect_j; f['fixed_ok']=(f.oracle_diff!=0)&(f.fixed_diff!=0)&(np.sign(f.oracle_diff)==np.sign(f.fixed_diff)); f['perfect_ok']=(f.oracle_diff!=0)&(f.perfect_diff!=0)&(np.sign(f.oracle_diff)==np.sign(f.perfect_diff)); stat=[]; perf=[]
 for key,g in f.groupby(['video_key','domain_id','context_id','span','category'],sort=True):
  if not isinstance(key,tuple): key=(key,)
  comp=int(((g.oracle_diff!=0)&(g.fixed_diff!=0)).sum())
  stat.append(dict({'analysis':'fixed_proxy',**dict(zip(['video_key','domain_id','context_id','span','category'],key)),'pairwise_ordering_accuracy':float(g.fixed_ok.sum()/comp) if comp else np.nan,'comparable_pairs':comp}))
  pc=int(((g.oracle_diff!=0)&(g.perfect_diff!=0)).sum())
  perf.append(dict({'analysis':'perfect_static_oracle',**dict(zip(['video_key','domain_id','context_id','span','category'],key)),'pairwise_ordering_accuracy':float(g.perfect_ok.sum()/pc) if pc else np.nan,'comparable_pairs':pc}))
 pd.DataFrame(stat).to_csv(out/'task_context_proxy_static_baseline.csv',index=False); pd.DataFrame(perf).to_csv(out/'task_context_proxy_perfect_static_reference.csv',index=False)
 # reversal labels: use existing detailed file if present
 rp=out/'task_context_proxy_reversal_detection.csv'; rev=pd.read_csv(rp,nrows=32390) if rp.exists() else None
 if rev is None or len(rev)!=32390:
  # derive reversals from original and conditioned rows using pair table and fixed proxy; pair table has original proxy via raw merge
  oo=raw[raw.context_id==0][['video_key','domain_id','global_index','oracle_signed_damage','proxy_signed_damage']].rename(columns={'oracle_signed_damage':'Dori','proxy_signed_damage':'Pori'}); cc=raw[raw.context_id>0][['video_key','domain_id','context_id','span','pair_index','frame_pair','class_name','class_index','category','global_index','oracle_signed_damage','proxy_signed_damage','Delta_model','unit_type','stage']]; rev=[]
  for (v,d,cid),g in cc.groupby(['video_key','domain_id','context_id'],sort=True):
   q=g.iloc[0]; z=g.merge(oo,on=['video_key','domain_id','global_index']); vals={int(r.global_index):r for r in z.itertuples(index=False)}
   for i,j in itertools.combinations(sorted(vals),2):
    x,y=vals[i],vals[j]; od=x.Dori-y.Dori; rd=x.oracle_signed_damage-y.oracle_signed_damage; op=x.Pori-y.Pori; rp0=x.proxy_signed_damage-y.proxy_signed_damage
    if min(abs(od),abs(rd),abs(op),abs(rp0))==0: continue
    rev.append({'video_key':v,'manifest_order':int(q.manifest_order),'class_name':str(q.class_name),'class_index':int(q.class_index),'domain_id':int(d),'category':str(q.category),'context_id':int(cid),'span':int(q.span),'pair_index':int(q.pair_index),'frame_pair':str(q.frame_pair),'global_index_i':i,'global_index_j':j,'pair_type':('A' if x.unit_type=='attention_head' else 'F')+('A' if y.unit_type=='attention_head' else 'F'),'D_ori_i':x.Dori,'D_ori_j':y.Dori,'D_rel_i':x.oracle_signed_damage,'D_rel_j':y.oracle_signed_damage,'D_proxy_ori_i':x.Pori,'D_proxy_ori_j':y.Pori,'D_proxy_rel_i':x.proxy_signed_damage,'D_proxy_rel_j':y.proxy_signed_damage,'oracle_original_diff':od,'oracle_conditioned_diff':rd,'proxy_original_diff':op,'proxy_conditioned_diff':rp0,'original_margin':abs(od),'conditioned_margin':abs(rd),'oracle_reversal':od*rd<0,'proxy_reversal':op*rp0<0,'Delta_model':float(q.Delta_model)})
  rev=pd.DataFrame(rev)
 rev.to_csv(rp,index=False); rs=pd.concat([pd.DataFrame([dict({'analysis':'overall'},**rev_counts(rev))]),rev_groups(rev,['category'],'category'),rev_groups(rev,['pair_type'],'pair_type'),rev_groups(rev,['span'],'span')],ignore_index=True); rs.to_csv(out/'task_context_proxy_reversal_summary.csv',index=False)
 # margins, span, severity
 rev['original_margin_quartile']=pd.qcut(rev.original_margin,4,labels=['Q0-Q25','Q25-Q50','Q50-Q75','Q75-Q100'],duplicates='drop'); rev['conditioned_margin_quartile']=pd.qcut(rev.conditioned_margin,4,labels=['Q0-Q25','Q25-Q50','Q50-Q75','Q75-Q100'],duplicates='drop'); pd.DataFrame([dict({'quartile':str(q),'axis':'original'},**rev_counts(g)) for q,g in rev.groupby('original_margin_quartile',dropna=False)]).to_csv(out/'task_context_proxy_margin_analysis.csv',index=False); pd.DataFrame([dict({'quartile':str(q),'axis':'conditioned','pairwise_ordering_accuracy':float((np.sign(g.proxy_conditioned_diff)==np.sign(g.oracle_conditioned_diff)).mean())},**rev_counts(g)) for q,g in rev.groupby('conditioned_margin_quartile',dropna=False)]).to_csv(out/'task_context_proxy_conditioned_margin.csv',index=False); pd.DataFrame([dict({'span':s},**metric_pairs(pair[pair.span==s]),**rev_counts(rev[rev.span==s])) for s in sorted(pair.span.unique())]).to_csv(out/'task_context_proxy_span_analysis.csv',index=False); rev['severity_quartile']=pd.qcut(rev.Delta_model,4,labels=['Q0-Q25','Q25-Q50','Q50-Q75','Q75-Q100'],duplicates='drop'); rev_groups(rev,['severity_quartile'],'severity').to_csv(out/'task_context_proxy_severity_analysis.csv',index=False)
 # type/stage/video/class
 add_pair_groups(pair,['pair_type'],'pair_type').to_csv(out/'task_context_proxy_type_summary.csv',index=False); add_pair_groups(pair,['stage_i'],'stage').to_csv(out/'task_context_proxy_stage_summary.csv',index=False); vv=add_pair_groups(pair,['video_key'],'video').merge(vm[['video_path','class_name']].rename(columns={'video_path':'video_key'}),on='video_key',how='left'); vv.to_csv(out/'task_context_proxy_video_summary.csv',index=False); vv.groupby('class_name',as_index=False).agg(pairwise_ordering_accuracy=('pairwise_ordering_accuracy','mean'),spearman=('spearman','mean'),kendall=('kendall','mean'),video_count=('video_key','nunique')).to_csv(out/'task_context_proxy_class_summary.csv',index=False)
 # same-span ranking-change agreement
 same=[]
 for (v,d,s),g in raw[raw.context_id>0].groupby(['video_key','domain_id','span'],sort=True):
  cids=sorted(g.context_id.unique()); by={c:g[g.context_id==c] for c in cids}; q=g.iloc[0]
  for c1,c2 in itertools.combinations(cids,2):
   a={int(r.global_index):r for r in by[c1].itertuples(index=False)}; b={int(r.global_index):r for r in by[c2].itertuples(index=False)}; comp=agree=changed=rec=0
   for i,j in itertools.combinations(sorted(a),2):
    so=np.sign(a[i].oracle_signed_damage-a[j].oracle_signed_damage); st=np.sign(b[i].oracle_signed_damage-b[j].oracle_signed_damage); po=np.sign(a[i].proxy_signed_damage-a[j].proxy_signed_damage); pt=np.sign(b[i].proxy_signed_damage-b[j].proxy_signed_damage)
    if 0 in (so,st,po,pt): continue
    comp+=1; oc=so!=st; pc=po!=pt; changed+=int(oc); rec+=int(oc and pc); agree+=int(oc==pc)
   same.append({'video_key':v,'class_name':str(q.class_name),'domain_id':int(d),'category':str(q.category),'span':int(s),'context_id_a':int(c1),'context_id_b':int(c2),'oracle_changed_pairs':changed,'proxy_reproduced_changed_pairs':rec,'comparable_pairs':comp,'context_pair_agreement':agree/comp if comp else np.nan})
 pd.DataFrame(same).to_csv(out/'task_context_proxy_same_span.csv',index=False)
 # bootstrap counts
 classes=sorted(vm.class_name.astype(str).unique()); counts={}
 for c in classes:
  g=pair[pair.class_name==c]; h=rev[rev.class_name==c]; fixed_ok=0; fixed_comp=0
  for key,z in f[f.class_name==c].groupby(['video_key','domain_id','context_id']): fixed_comp+=int(((z.oracle_diff!=0)&(z.fixed_diff!=0)).sum()); fixed_ok+=int(z.fixed_ok.sum())
  counts[c]=(int(g.strict_comparable.sum()),int(g.agreement.sum()),fixed_comp,fixed_ok,rev_counts(h))
 rng=np.random.default_rng(3407); boots=[]
 for b in range(10000):
  sm=rng.choice(classes,len(classes),replace=True); vals=[counts[str(c)] for c in sm]; cc=sum(x[0] for x in vals); ca=sum(x[1] for x in vals); fc=sum(x[2] for x in vals); fa=sum(x[3] for x in vals); tp=sum(x[4]['TP'] for x in vals); fp=sum(x[4]['FP'] for x in vals); fn=sum(x[4]['FN'] for x in vals); p=tp/(tp+fp) if tp+fp else np.nan; r=tp/(tp+fn) if tp+fn else np.nan; boots.append({'replicate':b,'conditioned_pairwise_accuracy':ca/cc if cc else np.nan,'fixed_pairwise_accuracy':fa/fc if fc else np.nan,'improvement_over_fixed':ca/cc-fa/fc if cc and fc else np.nan,'reversal_precision':p,'reversal_recall':r,'reversal_F1':2*p*r/(p+r) if np.isfinite(p) and np.isfinite(r) and p+r else np.nan})
 boot=pd.DataFrame(boots); boot.to_csv(out/'task_context_proxy_bootstrap.csv',index=False)
 # examples and optional audit
 ex=[]
 for tag,mask in [('A_oracle_reversal_recovered',(rev.oracle_reversal)&(rev.proxy_reversal)),('B_severity_matched_recovered',(rev.oracle_reversal)&(rev.proxy_reversal)&(rev.Delta_model>=rev.Delta_model.quantile(.5))),('D_oracle_reversal_missed',(rev.oracle_reversal)&(~rev.proxy_reversal)),('E_false_positive',(~rev.oracle_reversal)&(rev.proxy_reversal)),('F_stable_pair',(~rev.oracle_reversal)&(~rev.proxy_reversal))]:
  z=rev[mask].sort_values(['video_key','domain_id','span','pair_index','global_index_i','global_index_j']);
  if len(z): ex.append(dict({'example':tag},**z.iloc[0].to_dict()))
 if len(same): ex.append(dict({'example':'C_same_span_ranking_change_recovered'},**same[0]))
 pd.DataFrame(ex).to_csv(out/'task_context_proxy_examples.csv',index=False); pd.DataFrame([{'status':'NOT_RUN_OPTIONAL_NATIVE_AUTHORITATIVE','reason':'native tensor dot-gradient authoritative'}]).to_csv(out/'task_context_proxy_aligned_cf_audit.csv',index=False)
 caps=pd.concat([pd.read_csv(Path(x)/'task_context_proxy_capture_audit.csv') for x in [args.shard0,args.shard1]],ignore_index=True); checks=[('30_video_manifest_identity',30,vm.video_path.nunique()),('36_unit_identity',36,raw.global_index.nunique()),('21_context_identity',21,raw.context_id.nunique()),('checkpoint_sha256','4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63',json.load(open(Path(args.shard0)/'task_context_proxy_runtime_summary.json'))['checkpoint_sha256']),('raw_row_count',22680,len(raw)),('GPU_capture_contexts',630,caps[['video_key','context_id']].drop_duplicates().shape[0]),('native_shape_grad_equal',True,bool((caps.native_numel==caps.gradient_numel).all()))]; pd.DataFrame([{'check':k,'expected':e,'observed':o,'status':'PASS' if e==o else 'FAIL'} for k,e,o in checks]).to_csv(out/'task_context_proxy_identity_audit.csv',index=False)
 def ci(c): return [float(boot[c].quantile(.025)),float(boot[c].quantile(.975))]
 mean=float(boot.conditioned_pairwise_accuracy.mean()); fixed=float(boot.fixed_pairwise_accuracy.mean()); ci_cond=ci('conditioned_pairwise_accuracy'); ci_imp=ci('improvement_over_fixed'); support=ci_cond[0]>.5 and ci_imp[0]>0 and bool(rev.oracle_reversal.any() and rev.proxy_reversal.any()) and rev[rev.proxy_reversal].class_name.nunique()>=2 and raw[(raw.context_id>0)&(raw.unit_type=='attention_head')].shape[0]>0 and raw[(raw.context_id>0)&(raw.unit_type=='ffn_neuron')].shape[0]>0; decision='CONTEXTUAL_FIRST_ORDER_PROXY_PROMISING' if support else ('CONTEXTUAL_FIRST_ORDER_PROXY_REJECTED' if ci_cond[1]<.5 else 'CONTEXTUAL_FIRST_ORDER_PROXY_WEAK_OR_UNRESOLVED'); summary={'task':'task037_context_proxy','decision':decision,'raw_rows':len(raw),'videos':int(raw.video_key.nunique()),'classes':int(raw.class_name.nunique()),'domains':int(raw.domain_id.nunique()),'units':int(raw.global_index.nunique()),'contexts_per_video':21,'context_evaluations':630,'conditioned_pairwise_accuracy_mean':mean,'fixed_pairwise_accuracy_mean':fixed,'improvement_mean':mean-fixed,'bootstrap_ci':{'conditioned_pairwise_accuracy':ci_cond,'improvement_over_fixed':ci_imp,'reversal_precision':ci('reversal_precision'),'reversal_recall':ci('reversal_recall'),'reversal_F1':ci('reversal_F1')},'oracle_reversal_count':int(rev.oracle_reversal.sum()),'proxy_reversal_count':int(rev.proxy_reversal.sum()),'required_questions':{'A':bool(origv.pairwise_ordering_accuracy.mean()>.5),'B':bool(cv.pairwise_ordering_accuracy.mean()>.5),'C':bool(mean>fixed),'D':bool(rev.proxy_reversal.any()),'E':bool(rev[rev.category=='AA'].proxy_reversal.any()),'F':bool(rev[rev.category=='FF'].proxy_reversal.any()),'G':bool(rev[rev.category=='MIXED'].proxy_reversal.any()),'H':True,'I':True,'J':decision=='CONTEXTUAL_FIRST_ORDER_PROXY_PROMISING'}}; json.dump(summary,open(out/'task_context_proxy_summary.json','w'),indent=2); (out/'task_context_proxy_report.md').write_text('# Task037 Context-Conditioned First-Order Deletion Proxy Validation\n\nDecision: **'+decision+'**\n\nNative activation×gradient proxy; one backward per video-context; no masking rerun, calibration, pruning, or finetuning.\n\nRequired questions: '+json.dumps(summary['required_questions'],ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--shard0',required=True); p.add_argument('--shard1',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--video_manifest',required=True); p.add_argument('--confirmation_dir',required=True); p.add_argument('--unit_manifest',required=True); main(p.parse_args())


