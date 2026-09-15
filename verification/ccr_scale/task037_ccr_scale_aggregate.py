#!/usr/bin/env python3
from __future__ import annotations
import json, time, hashlib
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

EPS=1e-12; BOOT=10000; SEED=3407
SPANS=[1,2,4,8,16]

def norm(x):
 mn=x.min(dim=1,keepdim=True).values; mx=x.max(dim=1,keepdim=True).values; den=mx-mn; return torch.where(den.abs()<=EPS,torch.zeros_like(x),(x-mn)/(den+EPS)).clamp(0,1)

def stream_indices(vm,n):
 s=vm.sort_values(['within_class_position','class_index','manifest_order'])
 return s.head(n).index.tolist()

def select_indices(vm,setname): return vm.index[vm[setname].astype(bool)].tolist()

def choose_from_scores(scores, units, ascending=True):
 order=stable_order(scores,units,ascending=ascending); return int(units[order[0]])

def stable_order(scores, units, ascending=True):
 vals=scores.detach().cpu().tolist()
 if ascending:
  return sorted(range(len(units)), key=lambda j:(float(vals[j]), int(units[j])))
 return sorted(range(len(units)), key=lambda j:(-float(vals[j]), int(units[j])))

def main():
 ap=argparse.ArgumentParser(description='Aggregate TASK037 CCR scale shards')
 ap.add_argument('--output_dir', required=True)
 args=ap.parse_args()
 t0=time.time(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
 proxy=pd.concat([pd.read_csv(p) for p in sorted((out/'shards').glob('gpu*_proxy_raw.csv'))],ignore_index=True); oracle=pd.concat([pd.read_csv(p) for p in sorted((out/'shards').glob('gpu*_oracle_raw.csv'))],ignore_index=True)
 key=['video_key','context_id','domain_id','global_index']
 if proxy.duplicated(key).any() or oracle.duplicated(key).any(): raise AssertionError('duplicate raw keys')
 if set(map(tuple,proxy[key].to_numpy()))!=set(map(tuple,oracle[key].to_numpy())): raise AssertionError('proxy/oracle keys differ')
 merged=proxy.merge(oracle[key+['oracle_signed_damage','oracle_absolute_damage']],on=key,how='inner',validate='one_to_one')
 vm=pd.read_csv(out/'task_ccr_scale_video_manifest.csv'); vm['video_key']=vm.video_path.astype(str).str.split('/').str[-1]; vm_order={str(r.video_key):i for i,r in vm.iterrows()}
 unitm=pd.read_csv(out/'task_ccr_scale_unit_manifest.csv'); domcat=unitm[['domain_id','category']].drop_duplicates().set_index('domain_id').category.to_dict(); units_by={int(d):sorted(g.global_index.astype(int).tolist()) for d,g in unitm.groupby('domain_id')}
 if len(vm)!=30 or merged.video_key.nunique()!=30 or merged.context_id.nunique()!=11: raise AssertionError('raw completeness metadata')
 # Persist merged raw tables under the required names.
 proxy.to_csv(out/'task_ccr_scale_proxy_raw.csv',index=False); oracle.to_csv(out/'task_ccr_scale_oracle_raw.csv',index=False)
 domains=sorted(units_by); sets={s:select_indices(vm,s) for s in ['n9a','n9b','n9c']}; sets['n9a']=sorted(sets['n9a']); sets['n9b']=sorted(sets['n9b']); sets['n9c']=sorted(sets['n9c'])
 # index rows by frozen manifest order, not dataframe order
 vkeys=vm.sort_values('manifest_order').video_key.tolist(); vpos={k:i for i,k in enumerate(vkeys)}; classes=sorted(vm.class_index.unique()); cpos={c:i for i,c in enumerate(classes)}
 method_rows=[]; gap_rows=[]; decomp_rows=[]; o_rows=[]; p30_rows=[]; p9_rows=[]; n9rows=[]; type_rows=[]; curve_rows=[]; context_rows=[]; class_boot=[]
 all_gaps={'CCR':[],'frequency':[],'mean':[],'fixed':[]}; all_top={'CCR':[],'frequency':[],'mean':[],'fixed':[]}; domain_cats=[]
 per_domain_proxy_ccr={}; per_domain_oracle_ccr={}
 for d in domains:
  us=units_by[d]; n=len(us); base=merged[merged.domain_id==d].copy();
  P=torch.full((30,11,n),float('nan'),device='cuda',dtype=torch.float64); O=torch.full_like(P,float('nan'))
  ui={u:j for j,u in enumerate(us)}
  for r in base.itertuples(index=False):
   vi=vpos[str(r.video_key)]; j=ui[int(r.global_index)]; ci=int(r.context_id); P[vi,ci,j]=float(r.proxy_signed_damage); O[vi,ci,j]=float(r.oracle_signed_damage)
  if not torch.isfinite(P).all() or not torch.isfinite(O).all(): raise AssertionError(f'nonfinite/incomplete domain {d}')
  RP=norm(P.reshape(-1,n)).reshape(30,11,n); RO=norm(O.reshape(-1,n)).reshape(30,11,n); rp_video=RP[:,1:].max(1).values; ro_video=RO[:,1:].max(1).values
  ccr_p30=rp_video.mean(0); ccr_o30=ro_video.mean(0); per_domain_proxy_ccr[d]=ccr_p30; per_domain_oracle_ccr[d]=ccr_o30
  for j,u in enumerate(us):
   common={'domain_id':d,'category':domcat[d],'global_index':u,'proxy30_CCR':float(ccr_p30[j].cpu()),'oracle30_CCR':float(ccr_o30[j].cpu()),'oracle_rank':int(stable_order(ccr_o30,us).index(j)+1),'proxy30_rank':int(stable_order(ccr_p30,us).index(j)+1)}
   o_rows.append({k:common[k] for k in ['domain_id','category','global_index','oracle30_CCR','oracle_rank']}); p30_rows.append({k:common[k] for k in ['domain_id','category','global_index','proxy30_CCR','proxy30_rank']})
  n9scores={}
  for label,vidx in [('N9-A',sets['n9a']),('N9-B',sets['n9b']),('N9-C',sets['n9c'])]:
   ix=torch.tensor(vidx,device='cuda',dtype=torch.long); s=RP[ix,1:].max(1).values.mean(0); sel=choose_from_scores(s,us); best=int(us[stable_order(ccr_o30,us)[0]]); n9scores[label]=s
   n9rows.append({'replication':label,'domain_id':d,'category':domcat[d],'calibration_videos':len(vidx),'selected_unit':sel,'oracle_best_unit':best,'selected_oracle_CCR':float(ccr_o30[ui[sel]].cpu()),'oracle_best_CCR':float(ccr_o30.min().cpu()),'SelectionGap':float((ccr_o30[ui[sel]]-ccr_o30.min()).cpu()),'top1_oracle_identity':int(sel==best)})
  s9=n9scores['N9-A']; ix9=torch.tensor(sets['n9a'],device='cuda',dtype=torch.long); p9scores=s9; sel_ccr=choose_from_scores(s9,us)
  for j,u in enumerate(us): p9_rows.append({'domain_id':d,'category':domcat[d],'global_index':u,'proxy9_CCR':float(s9[j].cpu()),'proxy9_rank':int(stable_order(s9,us).index(j)+1)})
  # baseline aggregators use exactly the same N9-A videos and all 11 contexts
  sub=P[ix9]; winners=sub.argmin(dim=2); freq=torch.zeros(n,device='cuda',dtype=torch.float64)
  freq.scatter_add_(0,winners.reshape(-1),torch.ones(winners.numel(),device='cuda',dtype=torch.float64)); sel_freq=choose_from_scores(-freq,us) # highest count; global index tie via explicit scan below
  # explicit deterministic tie breaks
  sel_freq=sorted(us,key=lambda u:(-float(freq[ui[u]].cpu()),u))[0]
  mean_s=sub.mean(dim=(0,1)); sel_mean=sorted(us,key=lambda u:(float(mean_s[ui[u]].cpu()),u))[0]
  fixed_s=sub[:,0].mean(0); sel_fixed=sorted(us,key=lambda u:(float(fixed_s[ui[u]].cpu()),u))[0]
  best=float(ccr_o30.min().cpu()); bestu=us[stable_order(ccr_o30,us)[0]]
  for name,sel,score in [('CCR',sel_ccr,n9scores['N9-A']),('frequency',sel_freq,freq),('mean',sel_mean,mean_s),('fixed',sel_fixed,fixed_s)]:
   gap=float((ccr_o30[ui[sel]]-ccr_o30.min()).cpu()); row={'domain_id':d,'category':domcat[d],'method':name,'selected_unit':sel,'oracle_best_unit':bestu,'selected_oracle_CCR':float(ccr_o30[ui[sel]].cpu()),'oracle_best_CCR':best,'SelectionGap':gap,'top1_oracle_identity':int(sel==bestu)}; method_rows.append(row); gap_rows.append(row.copy()); all_gaps[name].append(gap); all_top[name].append(int(sel==bestu))
  decomp_rows.append({'domain_id':d,'category':domcat[d],'proxy30_selected_unit':int(us[stable_order(ccr_p30,us)[0]]),'proxy9_selected_unit':sel_ccr,'proxy30_vs_oracle_CCR_abs_mean':float((ccr_p30-ccr_o30).abs().mean().cpu()),'proxy9_vs_proxy30_selected_proxy_CCR_gap':float((n9scores['N9-A'][ui[sel_ccr]]-ccr_p30[ui[sel_ccr]]).cpu()),'end_to_end_oracle_SelectionGap':float((ccr_o30[ui[sel_ccr]]-ccr_o30.min()).cpu())})
  # calibration curve: N9-A is the frozen primary subset; others are deterministic stream prefixes.
  for N in [3,6,9,12,18,30]:
   vidx=sets['n9a'] if N==9 else [vpos[k] for k in vm.iloc[stream_indices(vm,N)].video_key.tolist()]; ix=torch.tensor(vidx,device='cuda',dtype=torch.long); ss=RP[ix,1:].max(1).values.mean(0); sel=choose_from_scores(ss,us); curve_rows.append({'curve':'calibration_N','N':N,'domain_id':d,'category':domcat[d],'selected_unit':sel,'oracle_best_unit':bestu,'SelectionGap':float((ccr_o30[ui[sel]]-ccr_o30.min()).cpu()),'top1_oracle_identity':int(sel==bestu)})
  for M,ids in [(5,[1,3,5,7,9]),(10,list(range(1,11)))]:
   ss=RP[ix9][:,ids].max(1).values.mean(0); sel=choose_from_scores(ss,us); context_rows.append({'context_count':M,'context_ids':'|'.join(map(str,ids)),'domain_id':d,'category':domcat[d],'selected_unit':sel,'oracle_best_unit':bestu,'SelectionGap':float((ccr_o30[ui[sel]]-ccr_o30.min()).cpu()),'top1_oracle_identity':int(sel==bestu)})
  # class bootstrap contributions (max relation per video, then class mean)
  cls=torch.zeros((10,n),device='cuda',dtype=torch.float64)
  for ci,c in enumerate(classes):
   vids=[vpos[k] for k in vm[vm.class_index==c].sort_values('manifest_order').video_key.tolist()]; cls[ci]=rp_video[vids].mean(0)
  gen=torch.Generator(device='cuda'); gen.manual_seed(SEED); sample=torch.randint(0,10,(BOOT,10),device='cuda',generator=gen); w=torch.zeros((BOOT,10),device='cuda',dtype=torch.float64); w.scatter_add_(1,sample,torch.ones_like(sample,dtype=torch.float64)); bs=(w@cls)/10.0; chosen=bs.argmin(1); gaps=ccr_o30[chosen]-ccr_o30.min(); agree=(chosen==ccr_o30.argmin()).to(torch.int8); class_boot.append(pd.DataFrame({'domain_id':d,'category':domcat[d],'replicate':np.arange(BOOT),'CCR_proxy_class_bootstrap_SelectionGap':gaps.cpu().numpy(),'proxy_oracle_top1_agreement':agree.cpu().numpy()}))
  domain_cats.append(domcat[d])
  for name in ['CCR','frequency','mean','fixed']:
   pass
 # summary tables
 pd.DataFrame(o_rows).to_csv(out/'task_ccr_scale_oracle_ccr.csv',index=False); pd.DataFrame(p30_rows).to_csv(out/'task_ccr_scale_proxy30_ccr.csv',index=False)
 pd.DataFrame(p9_rows).to_csv(out/'task_ccr_scale_proxy9_ccr.csv',index=False); pd.DataFrame(decomp_rows).to_csv(out/'task_ccr_scale_error_decomposition.csv',index=False); pd.DataFrame(n9rows).to_csv(out/'task_ccr_scale_n9_replications.csv',index=False); pd.DataFrame(method_rows).to_csv(out/'task_ccr_scale_method_comparison.csv',index=False); pd.DataFrame(gap_rows).to_csv(out/'task_ccr_scale_selection_gap.csv',index=False)
 for cat in ['AA','FF','MIXED']:
  rows=[r for r in method_rows if r['category']==cat]; c=[r for r in rows if r['method']=='CCR']; f=[r for r in rows if r['method']=='frequency']; m=[r for r in rows if r['method']=='mean']; x=[r for r in rows if r['method']=='fixed'];
  type_rows.append({'category':cat,'domains':len(c),'CCR_mean_SelectionGap':np.mean([r['SelectionGap'] for r in c]) if c else np.nan,'CCR_median_SelectionGap':np.median([r['SelectionGap'] for r in c]) if c else np.nan,'CCR_q25':np.quantile([r['SelectionGap'] for r in c],.25) if c else np.nan,'CCR_q75':np.quantile([r['SelectionGap'] for r in c],.75) if c else np.nan,'CCR_max':max([r['SelectionGap'] for r in c],default=np.nan),'CCR_top1_rate':np.mean([r['top1_oracle_identity'] for r in c]) if c else np.nan,'frequency_mean_SelectionGap':np.mean([r['SelectionGap'] for r in f]) if f else np.nan,'mean_mean_SelectionGap':np.mean([r['SelectionGap'] for r in m]) if m else np.nan,'fixed_mean_SelectionGap':np.mean([r['SelectionGap'] for r in x]) if x else np.nan})
 pd.DataFrame(type_rows).to_csv(out/'task_ccr_scale_type_summary.csv',index=False); pd.DataFrame(curve_rows).to_csv(out/'task_ccr_scale_calibration_curve.csv',index=False); pd.DataFrame(context_rows).to_csv(out/'task_ccr_scale_context_curve.csv',index=False)
 # Domain bootstrap preserving category counts, fully tensorized on CUDA.
 gen=torch.Generator(device='cuda'); gen.manual_seed(SEED); db={}
 for cat in ['AA','FF','MIXED']:
  ids=[i for i,x in enumerate(domain_cats) if x==cat]; vals={m:torch.tensor([all_gaps[m][i] for i in ids],device='cuda',dtype=torch.float64) for m in all_gaps}; samp=torch.randint(0,len(ids),(BOOT,len(ids)),device='cuda',generator=gen); db[cat]={m:vals[m][samp].mean(1) for m in vals}; db[cat]['CCR_top1_rate']=torch.tensor([all_top['CCR'][i] for i in ids],device='cuda',dtype=torch.float64)[samp].mean(1)
 # Preserve the observed category counts by resampling within each category,
 # then average the three category means for every bootstrap replicate.
 cat_list=['AA','FF','MIXED']; cat_counts=torch.tensor([sum(x==c for x in domain_cats) for c in cat_list],device='cuda',dtype=torch.float64); cat_weights=cat_counts/cat_counts.sum()
 comb={m:(torch.stack([db[c][m] for c in cat_list],dim=1)*cat_weights.view(1,-1)).sum(1) for m in all_gaps}; top=(torch.stack([db[c]['CCR_top1_rate'] for c in cat_list],dim=1)*cat_weights.view(1,-1)).sum(1); dom_boot=pd.DataFrame({'replicate':np.arange(BOOT),'CCR_minus_frequency':(comb['CCR']-comb['frequency']).cpu().numpy(),'CCR_minus_mean':(comb['CCR']-comb['mean']).cpu().numpy(),'CCR_minus_fixed':(comb['CCR']-comb['fixed']).cpu().numpy(),'CCR_top1_rate':top.cpu().numpy()}); dom_boot.to_csv(out/'task_ccr_scale_domain_bootstrap.csv',index=False)
 cb=pd.concat(class_boot,ignore_index=True); cb.to_csv(out/'task_ccr_scale_class_bootstrap.csv',index=False)
 # CCR-vs-frequency delta and method summary.
 deltas=[]
 for d in domains:
  c=next(r for r in method_rows if r['domain_id']==d and r['method']=='CCR'); f=next(r for r in method_rows if r['domain_id']==d and r['method']=='frequency'); deltas.append({'domain_id':d,'category':domcat[d],'CCR_SelectionGap':c['SelectionGap'],'frequency_SelectionGap':f['SelectionGap'],'DeltaGap_CCR_minus_frequency':c['SelectionGap']-f['SelectionGap'],'winner':'CCR' if c['SelectionGap']<f['SelectionGap']-EPS else ('frequency' if f['SelectionGap']<c['SelectionGap']-EPS else 'tie')})
 pd.DataFrame(deltas).to_csv(out/'task_ccr_scale_ccr_frequency_delta.csv',index=False)
 # Efficiency and runtime aggregation.
 runt=[]
 for p in sorted((out/'shards').glob('gpu*_runtime.json')): runt.append(json.loads(p.read_text()))
 gpu_peak=max([float(x.get('gpu_peak_memory_mb',0)) for x in runt] or [0]); runtime={'gpu_used':True,'gpu0_workload':'proxy + exact masked oracle on even manifest_order videos','gpu1_workload':'proxy + exact masked oracle on odd manifest_order videos','gpu_peak_memory_mb':gpu_peak,'cpu_only_workload':'CSV I/O, manifest joins, SHA256, report writing','number_forward_passes':int(sum(x.get('number_forward_passes',0) for x in runt)),'number_backward_passes':int(sum(x.get('number_backward_passes',0) for x in runt)),'number_masked_unit_cases':int(sum(x.get('number_masked_unit_cases',0) for x in runt)),'effective_mask_batch_size':int(min([x.get('effective_mask_batch_size',1) for x in runt] or [1])),'proxy_wall_clock_seconds':float(max([x.get('wall_clock_seconds',0) for x in runt] or [0])),'oracle_wall_clock_seconds':float(max([x.get('wall_clock_seconds',0) for x in runt] or [0])),'aggregation_bootstrap_wall_clock_seconds':time.time()-t0,'wall_clock_seconds':time.time()-t0,'shards':runt,'proxy_only_estimated_N9_M5_cost_seconds':float(sum(x.get('wall_clock_seconds',0) for x in runt)*9*5/(30*10*11)),'proxy_only_estimated_N9_M10_cost_seconds':float(sum(x.get('wall_clock_seconds',0) for x in runt)*9*10/(30*10*11))}
 runtime['wall_clock_seconds']=float(max([x.get('wall_clock_seconds',0) for x in runt] or [0]) + runtime['aggregation_bootstrap_wall_clock_seconds'])
 (out/'task_ccr_scale_runtime.json').write_text(json.dumps(runtime,indent=2,allow_nan=False)+'\n')
 eff=pd.DataFrame([{'component':'proxy_construction','wall_clock_seconds':runtime['proxy_wall_clock_seconds']},{'component':'oracle_construction','wall_clock_seconds':runtime['oracle_wall_clock_seconds']},{'component':'CCR_aggregation_and_statistics','wall_clock_seconds':runtime['aggregation_bootstrap_wall_clock_seconds']},{'component':'practical_proxy_only_N9_M5_estimate','wall_clock_seconds':runtime['proxy_only_estimated_N9_M5_cost_seconds']},{'component':'practical_proxy_only_N9_M10_estimate','wall_clock_seconds':runtime['proxy_only_estimated_N9_M10_cost_seconds']}]); eff.to_csv(out/'task_ccr_scale_efficiency.csv',index=False)
 def ci(x): return [float(np.quantile(x,.025)),float(np.quantile(x,.975))]
 summary={'task':'TASK037_CCR_SCALE','decision':'DIAGNOSTIC_ONLY','new_domains':len(domains),'category_counts':{c:domain_cats.count(c) for c in ['AA','FF','MIXED']},'selected_units':len(unitm),'videos':30,'contexts_per_video':11,'relation_contexts_for_CCR':10,'primary_N9':'N9-A','domain_bootstrap_CI_95':{'CCR_minus_frequency':ci(dom_boot.CCR_minus_frequency),'CCR_minus_mean':ci(dom_boot.CCR_minus_mean),'CCR_minus_fixed':ci(dom_boot.CCR_minus_fixed),'CCR_top1_rate':ci(dom_boot.CCR_top1_rate)},'primary_results':type_rows,'CCR_frequency_wins_ties':{'CCR':sum(x['winner']=='CCR' for x in deltas),'ties':sum(x['winner']=='tie' for x in deltas),'frequency':sum(x['winner']=='frequency' for x in deltas)},'CCR_frequency_delta_mean':float(np.mean([x['DeltaGap_CCR_minus_frequency'] for x in deltas])),'CCR_frequency_delta_median':float(np.median([x['DeltaGap_CCR_minus_frequency'] for x in deltas])),'exact_oracle_validation_only':True,'final_method_requires_oracle':False,'no_pruning':True,'no_finetuning':True}
 (out/'task_ccr_scale_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 method_df=pd.DataFrame(method_rows)
 overall_rows=[]
 for method in ['CCR','frequency','mean','fixed']:
  z=method_df[method_df.method==method]
  overall_rows.append({'method':method,'mean_SelectionGap':float(z.SelectionGap.mean()),'median_SelectionGap':float(z.SelectionGap.median()),'q25':float(z.SelectionGap.quantile(.25)),'q75':float(z.SelectionGap.quantile(.75)),'max':float(z.SelectionGap.max()),'oracle_top1_rate':float(z.top1_oracle_identity.mean())})
 n9_df=pd.DataFrame(n9rows); n9_summary=n9_df.groupby('replication',as_index=False).agg(top1_oracle_rate=('top1_oracle_identity','mean'),mean_SelectionGap=('SelectionGap','mean'),median_SelectionGap=('SelectionGap','median'))
 cal_df=pd.DataFrame(curve_rows); cal_summary=cal_df.groupby('N',as_index=False).agg(mean_SelectionGap=('SelectionGap','mean'),median_SelectionGap=('SelectionGap','median'),top1_oracle_rate=('top1_oracle_identity','mean'))
 ctx_df=pd.DataFrame(context_rows); ctx_summary=ctx_df.groupby('context_count',as_index=False).agg(mean_SelectionGap=('SelectionGap','mean'),median_SelectionGap=('SelectionGap','median'),top1_oracle_rate=('top1_oracle_identity','mean'))
 lines=['# TASK037 CCR independent domain-scale and calibration-efficiency validation','',f"Decision: DIAGNOSTIC_ONLY. New domains: {len(domains)} (AA={domain_cats.count('AA')}, FF={domain_cats.count('FF')}, MIXED={domain_cats.count('MIXED')}; only 9 eligible MIXED domains remained after excluding the old 9, so no category substitution was made).",'',f"Primary N9-A uses the frozen 9-video manifest; exact oracle is validation-only. Selected units: {len(unitm)}; contexts: 10 relation + original.",'', '## Overall domain-scale results','',pd.DataFrame(overall_rows).to_markdown(index=False),'', '## Type-stratified results','',pd.DataFrame(type_rows).to_markdown(index=False),'', '## CCR versus frequency', '', f"CCR wins={sum(x['winner']=='CCR' for x in deltas)}, ties={sum(x['winner']=='tie' for x in deltas)}, frequency wins={sum(x['winner']=='frequency' for x in deltas)}; mean DeltaGap={np.mean([x['DeltaGap_CCR_minus_frequency'] for x in deltas]):.6g}; median={np.median([x['DeltaGap_CCR_minus_frequency'] for x in deltas]):.6g}.",'', '## Domain bootstrap (95% CI)','',json.dumps(summary['domain_bootstrap_CI_95'],indent=2),'', '## N=9 replication stability','',n9_summary.to_markdown(index=False),'', '## Calibration-size curve','',cal_summary.to_markdown(index=False),'', '## Context-count curve','',ctx_summary.to_markdown(index=False),'', '## Class bootstrap (95% CI)','',f"CCR proxy class-bootstrap SelectionGap CI: {ci(cb.CCR_proxy_class_bootstrap_SelectionGap)}; proxy/oracle top-1 agreement CI: {ci(cb.proxy_oracle_top1_agreement)}.",'', '## Resource audit','', 'The experiment used CUDA on GPU0/GPU1. Batch-4 and batch-8 masked preflight exceeded the 1e-6 exactness tolerance, so the registered fallback effective mask batch size was 1. The exact masked oracle was used only to validate the proxy and CCR selection; it is not part of the eventual pruning method. No pruning, finetuning, BMS changes, CCR redesign, or final-N tuning was performed.', '', 'Efficiency:','',eff.to_markdown(index=False),'', '```json',json.dumps(runtime,indent=2), '```']
 (out/'task_ccr_scale_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
