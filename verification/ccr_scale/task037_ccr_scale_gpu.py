#!/usr/bin/env python3
"""GPU runner for the independent CCR scale audit (proxy + exact oracle)."""
from __future__ import annotations
import argparse, contextlib, json, time
from pathlib import Path
import sys
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task037_context_proxy_gpu import (
    SelectedUnit, ensure_project_importable, set_seed, resolve_device, sha256_file,
    git_value, make_checkpoint_identity, unwrap_logits, infer_temporal_length,
    _native_slice,
)

def parse_args():
 p=argparse.ArgumentParser()
 for n in ['project_root','checkpoint','output_dir','val_list','unit_manifest','domain_manifest','video_manifest','context_manifest']:
  p.add_argument('--'+n,required=True)
 p.add_argument('--device',default='cuda:0'); p.add_argument('--adapter',default='ucf101_videoswin_probe_adapter_v2'); p.add_argument('--frame_root',default=''); p.add_argument('--num_classes',type=int,default=10); p.add_argument('--videos_per_class',type=int,default=3); p.add_argument('--num_workers',type=int,default=2); p.add_argument('--seed',type=int,default=3407); p.add_argument('--shard',type=int,choices=[0,1],required=True); p.add_argument('--mask_batch_size',type=int,default=4); p.add_argument('--max_videos',type=int,default=0)
 return p.parse_args()

def contexts(path, actual_t, core):
 cm=pd.read_csv(path); expected={(s,p) for s in (1,2,4,8,16) for p in (0,8)}
 got={(int(r.span),int(r.canonical_pair_index)) for r in cm.itertuples(index=False) if str(r.context_kind)=='relation'}
 if len(cm)!=11 or got!=expected: raise AssertionError('scale context manifest identity')
 allit=core.enumerate_fixed_cardinality_temporal_pairs(actual_t); by={(int(i.block_size),int(i.pair_index)):i for i in allit}; out=[]
 for r in cm.sort_values('context_id').itertuples(index=False):
  if str(r.context_kind)=='original': out.append({'context_id':0,'span':0,'pair_index':-1,'frame_pair':'original','intervention':None}); continue
  k=(int(r.span),int(r.canonical_pair_index)); it=by.get(k)
  if it is None or (int(it.left_start),int(it.right_start))!=(int(r.frame_a),int(r.frame_b)): raise AssertionError(f'pair mismatch {k}')
  out.append({'context_id':int(r.context_id),'span':k[0],'pair_index':k[1],'frame_pair':f'{int(r.frame_a)}-{int(r.frame_b)}','intervention':it})
 return out

def load_units(path, specs):
 um=pd.read_csv(path); by={str(s.name):s for s in specs}; out=[]
 for r in um.sort_values('global_index').itertuples(index=False):
  typ='head' if str(r.unit_type)=='attention_head' else 'neuron'; spec=by.get(str(r.layer))
  if spec is None or spec.unit_type!=typ: raise AssertionError(f'unit mismatch {r.layer}/{typ}')
  ui=int(r.unit_index)
  if not (0<=ui<int(spec.num_units)): raise AssertionError('unit index range')
  out.append(SelectedUnit(int(r.global_index),str(r.layer),typ,ui,spec))
 if len({u.global_index for u in out})!=len(out): raise AssertionError('duplicate selected units')
 return out

@contextlib.contextmanager
def temporary_batch_masks(units):
 """Apply one whole-unit mask to each sample in a batch, restoring hooks exactly."""
 by={}
 for j,u in enumerate(units): by.setdefault(id(u.spec.hook_module),[]).append((j,u))
 handles=[]
 try:
  for key,items in by.items():
   spec=items[0][1].spec
   def make(sp, pairs):
    def hook(_module, inputs):
     values=inputs[0].clone()
     for j,u in pairs:
      if sp.unit_type=='head':
       hd=int(sp.module.head_dim); h=int(sp.num_units)
       if values.shape[-1]!=h*hd: raise RuntimeError(f'bad attention shape {tuple(values.shape)}')
       z=values[j].reshape(*values[j].shape[:-1],h,hd); z[...,int(u.unit_index),:]=0.0; values[j]=z.reshape_as(values[j])
      elif sp.unit_type=='neuron': values[j,...,int(u.unit_index)]=0.0
      else: raise ValueError(sp.unit_type)
     return (values,)+tuple(inputs[1:])
    return hook
   handles.append(spec.hook_module.register_forward_pre_hook(make(spec,items)))
  yield
 finally:
  for h in handles: h.remove()

@contextlib.contextmanager
def temporary_one_mask(unit):
 with temporary_batch_masks([unit]): yield

def true_logits(model, videos):
 with torch.no_grad(): return unwrap_logits(model(videos))

def masked_values(model, inp, label, units, batch_size):
 vals=[]
 for start in range(0,len(units),batch_size):
  group=units[start:start+batch_size]; batch=inp.repeat(len(group),1,1,1,1)
  with temporary_batch_masks(group):
   with torch.no_grad(): logits=unwrap_logits(model(batch))
  vals.extend(logits[:,label].detach().to(torch.float64).cpu().tolist()); del batch,logits
 return vals

def sequential_values(model, inp, label, units):
 vals=[]
 for u in units:
  with temporary_one_mask(u):
   with torch.no_grad(): z=unwrap_logits(model(inp))
  vals.append(float(z[0,label].detach().to(torch.float64).cpu().item()))
 return vals

def preflight(model, inp, label, units):
 chosen=units[:8]
 if not ({u.unit_type for u in chosen}=={'head','neuron'}):
  # deterministic first eight covering both types
  chosen=units[:]
  hs=next((u for u in units if u.unit_type=='head'),None); ns=next((u for u in units if u.unit_type=='neuron'),None)
  chosen=[hs,ns]+[u for u in units if u not in (hs,ns)][:6]
 seq=sequential_values(model,inp,label,chosen); selected_bs=1; diffs={}
 for bs in (4,8):
  try:
   bat=masked_values(model,inp,label,chosen,bs); diff=max(abs(a-b) for a,b in zip(seq,bat)); order_seq=sorted(range(len(seq)),key=lambda i:seq[i]); order_bat=sorted(range(len(bat)),key=lambda i:bat[i]);
   margins=sorted(seq); ok=bool(torch.isfinite(torch.tensor(bat)).all()) and diff<=1e-6 and (order_seq==order_bat or min(abs(margins[i+1]-margins[i]) for i in range(len(margins)-1))<=1e-7)
   diffs[str(bs)]={'max_abs_true_class_diff':diff,'ordering_identity':order_seq==order_bat,'pass':ok}
   if ok: selected_bs=bs
  except RuntimeError as e:
   if 'out of memory' in str(e).lower(): torch.cuda.empty_cache(); diffs[str(bs)]={'pass':False,'oom':True}
   else: raise
 # restoration exact
 before=float(true_logits(model,inp)[0,label].detach().to(torch.float64).cpu().item()); sequential_values(model,inp,label,chosen); after=float(true_logits(model,inp)[0,label].detach().to(torch.float64).cpu().item())
 if before!=after: raise AssertionError('mask restoration not exact')
 return selected_bs,diffs

def run(a):
 t0=time.time(); project=Path(a.project_root).resolve(); ck=Path(a.checkpoint).resolve(); out=Path(a.output_dir).resolve(); (out/'shards').mkdir(parents=True,exist_ok=True); ensure_project_importable(project); set_seed(a.seed); dev=resolve_device(a.device)
 core=__import__('task040_htor_core'); ctfrs=__import__('probe_ctfrs_dynamic_function'); adapter=__import__(a.adapter); model,meta=adapter.build_model_for_probe(checkpoint=str(ck),device=dev); model.eval(); [p.requires_grad_(False) for p in model.parameters()]
 specs=ctfrs.discover_unit_layers(model); units=load_units(a.unit_manifest,specs); contexts_list=None
 loader,selected_indices,chosen_classes=ctfrs.build_balanced_loader(project_root=project,val_list=a.val_list,frame_root=a.frame_root,num_classes=a.num_classes,videos_per_class=a.videos_per_class,num_workers=a.num_workers,seed=a.seed)
 vm=pd.read_csv(a.video_manifest); vm_by={str(r.video_path):r for r in vm.itertuples(index=False)}; unit_dom={int(r.global_index):int(r.domain_id) for r in pd.read_csv(a.unit_manifest).itertuples(index=False)}; ds=loader.dataset
 while hasattr(ds,'dataset'): ds=ds.dataset
 first=next(iter(loader)); actual_t=infer_temporal_length(first[0]); contexts_list=contexts(a.context_manifest,actual_t,core)
 # ensure GPU sharding from frozen manifest; val_list should already match but explicit check
 shard_rows=vm[vm.gpu_shard==a.shard]
 if len(shard_rows)!=15: raise AssertionError(f'shard {a.shard} expected 15')
 pf={}; pf_bs=8
 for pc in contexts_list[:2]:
  pbase=first[0][0].float().to(dev)
  pinp=pbase.unsqueeze(0).clone() if pc['intervention'] is None else core.apply_temporal_interventions(pbase,[pc['intervention']],time_dim=1)[0]
  if pinp.ndim==4: pinp=pinp.unsqueeze(0)
  bs_one,pf_one=preflight(model,pinp.float().to(dev),int(first[1][0].item()),units); pf[str(pc['context_id'])]=pf_one; pf_bs=min(pf_bs,bs_one)
 rows_p=[]; rows_o=[]; nforward=0; nback=0; masked_cases=0; videos_done=0
 for batch in loader:
  if a.max_videos and videos_done>=a.max_videos: break
  videos,targets,indices=batch[:3]; dsidx=int(indices[0].item()); vid=str(ds.clips[dsidx][0]); key=Path(vid).name
  if key not in vm_by: raise AssertionError(f'video missing {key}')
  vr=vm_by[key]
  if int(vr.gpu_shard)!=a.shard: continue
  label=int(targets[0].item()); base=videos[0].float().to(dev,non_blocking=True).detach()
  for c in contexts_list:
   inp=base.unsqueeze(0).clone() if c['intervention'] is None else core.apply_temporal_interventions(base,[c['intervention']],time_dim=1)[0]
   if inp.ndim==4: inp=inp.unsqueeze(0)
   inp=inp.float().to(dev,non_blocking=True); inp.requires_grad_(True); model.zero_grad(set_to_none=True); captures={}; hs=[]
   # hooks installed locally; capture every unique selected layer
   by={id(u.spec.hook_module):u.spec for u in units}
   for k,sp in by.items():
    def make(cap,sp):
     def hook(_m,inputs):
      x=inputs[0]; x.retain_grad(); cap[id(sp.hook_module)]=x
     return hook
    hs.append(sp.hook_module.register_forward_pre_hook(make(captures,sp)))
   logits=unwrap_logits(model(inp)); target=logits[0,label]; target.backward(); nforward+=1; nback+=1
   for h in hs: h.remove()
   for u in units:
    x=captures[id(u.spec.hook_module)]; g=x.grad; signed=float((_native_slice(x,u.spec,u.unit_index)*_native_slice(g,u.spec,u.unit_index)).sum().detach().to(torch.float64).cpu().item())
    common={'class_name':str(vr.class_name),'class_index':int(vr.class_index),'video_id':vid,'video_key':key,'manifest_order':int(vr.manifest_order),'context_id':int(c['context_id']),'span':int(c['span']),'pair_index':int(c['pair_index']),'frame_pair':str(c['frame_pair']),'domain_id':unit_dom[u.global_index],'global_index':u.global_index,'unit_type':'attention_head' if u.unit_type=='head' else 'ffn_neuron','layer':u.layer_name,'unit_index':u.unit_index,'true_class_logit':float(target.detach().cpu().item()),'proxy_signed_damage':signed,'proxy_absolute_damage':abs(signed)}
    rows_p.append(common)
   # exact oracle using the same unmasked baseline from this context
   unmasked=float(target.detach().to(torch.float64).cpu().item()); masked=masked_values(model,inp.detach(),label,units,pf_bs); nforward += (len(units)+pf_bs-1)//pf_bs; masked_cases+=len(units)
   for u,mv in zip(units,masked):
    rr=next(r for r in rows_p[::-1] if int(r['global_index'])==u.global_index and int(r['context_id'])==int(c['context_id']) and r['video_key']==key)
    oo={k:rr[k] for k in ('class_name','class_index','video_id','video_key','manifest_order','context_id','span','pair_index','frame_pair','domain_id','global_index','unit_type','layer','unit_index','true_class_logit')}; oo.update({'oracle_signed_damage':unmasked-float(mv),'oracle_absolute_damage':abs(unmasked-float(mv))}); rows_o.append(oo)
   del logits,target,inp,captures
  videos_done+=1
 fields_p=['class_name','class_index','video_id','video_key','manifest_order','context_id','span','pair_index','frame_pair','domain_id','global_index','unit_type','layer','unit_index','true_class_logit','proxy_signed_damage','proxy_absolute_damage']; fields_o=['class_name','class_index','video_id','video_key','manifest_order','context_id','span','pair_index','frame_pair','domain_id','global_index','unit_type','layer','unit_index','true_class_logit','oracle_signed_damage','oracle_absolute_damage']
 pd.DataFrame(rows_p,columns=fields_p).to_csv(out/'shards'/f'gpu{a.shard}_proxy_raw.csv',index=False); pd.DataFrame(rows_o,columns=fields_o).to_csv(out/'shards'/f'gpu{a.shard}_oracle_raw.csv',index=False)
 identity=make_checkpoint_identity(model,ck,meta,specs); identity.update({'task':'TASK037_CCR_SCALE','device':str(dev),'checkpoint_sha256':sha256_file(ck),'expected_checkpoint_sha256':'4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63','shard':a.shard,'videos_processed':videos_done,'contexts_per_video':len(contexts_list),'context_evaluations':videos_done*len(contexts_list),'selected_unit_count':len(units),'proxy_rows':len(rows_p),'oracle_rows':len(rows_o),'number_forward_passes':nforward,'number_backward_passes':nback,'number_masked_unit_cases':masked_cases,'effective_mask_batch_size':pf_bs,'preflight':pf,'gpu_peak_memory_mb':torch.cuda.max_memory_allocated(dev)/1024**2,'wall_clock_seconds':time.time()-t0,'no_physical_pruning':True,'no_finetuning':True})
 Path(out/'shards'/f'gpu{a.shard}_runtime.json').write_text(json.dumps(identity,indent=2,allow_nan=False)+'\n')
 print(json.dumps(identity,indent=2))
if __name__=='__main__': run(parse_args())
