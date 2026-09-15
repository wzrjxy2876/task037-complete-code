#!/usr/bin/env python3
"""Type-stratified follow-up for the Task037 motivation audit.

Pairs are selected only from the existing offline table.  The selection is
rank-based within AA, FF, and AF categories; no post-hoc masking result enters
the manifest.  GPU masking is delegated to the already validated diagnostic
runner, which is imported without touching production pruning code.
"""
from __future__ import annotations
import csv, json, shutil, sys
from pathlib import Path


def read(path):
    with Path(path).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def write(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else ['status']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def typ(r):
    a,b=r['unit_type_i'],r['unit_type_j']
    return 'FF' if a.startswith('ffn') and b.startswith('ffn') else ('AA' if a.startswith('attention') and b.startswith('attention') else 'AF')

def main():
    src=Path('/data/jixinye25/work1/output/task037_temporal_redundancy_motivation')
    out=Path('/data/jixinye25/work1/output/task037_temporal_substitutability_type_validation')
    out.mkdir(parents=True,exist_ok=True)
    rows=read(src/'task037_temporal_gap_global_pairs.csv')
    selected=[]; summary=[]
    for category in ('FF','AA','AF'):
        rr=[r for r in rows if typ(r)==category]
        rr=sorted(rr,key=lambda r:(-float(r['A_global']),int(r['global_index_i']),int(r['global_index_j'])))
        top=rr[:50]
        if len(top)<2:
            summary.append(dict(category=category,eligible_pairs=len(rr),status='INSUFFICIENT')); continue
        mot=max(top,key=lambda r:(float(r['local_std']),-int(r['global_index_i']),-int(r['global_index_j'])))
        rank=top.index(mot); lo=(rank//10)*10; block=top[lo:lo+10]
        ctrl=min([r for r in block if r['pair_id']!=mot['pair_id']],key=lambda r:(float(r['local_std']),-float(r['A_global']),int(r['global_index_i'])))
        for cohort,r in (('motivation',mot),('consistent_control',ctrl)):
            x=dict(r); x['cohort']=cohort; x['pair_type']=category; x['global_rank_top50']=rr.index(r)+1; x['matched_rank_block']=f'{lo+1}-{lo+len(block)}'; selected.append(x)
        summary.append(dict(category=category,eligible_pairs=len(rr),status='SELECTED',motivation_pair=mot['pair_id'],control_pair=ctrl['pair_id'],motivation_A_global=mot['A_global'],control_A_global=ctrl['A_global'],global_gap=abs(float(mot['A_global'])-float(ctrl['A_global'])),motivation_local_std=mot['local_std'],control_local_std=ctrl['local_std'],domains=f"{mot['domain_id']};{ctrl['domain_id']}"))
    write(out/'task037_temporal_substitutability_type_pairs.csv',selected)
    write(out/'task037_temporal_substitutability_type_summary.csv',summary)
    # Reuse exact local curves and frozen field identities for selected pairs.
    local=read(src/'task037_temporal_gap_local_similarity.csv'); ids={r['pair_id'] for r in selected}
    write(out/'task037_temporal_gap_local_similarity.csv',[r for r in local if r['pair_id'] in ids])
    # Rebuild a fresh frozen manifest for this type-stratified cohort from the
    # same authoritative unit mapping; no previous masking result is consulted.
    map_rows=read('/home/jixinye25/jxy_work1/swintrans_task037/task014_n09/contribution_unit_mapping.csv')
    manifest=[]
    for r in selected:
        for side,gidx in (('i',int(r['global_index_i'])),('j',int(r['global_index_j']))):
            u=map_rows[gidx]
            manifest.append(dict(pair_id=r['pair_id'],cohort=r['cohort'],pair_side=side,domain_id=r['domain_id'],global_index=gidx,layer=u['layer'],unit_type=u['unit_type'],unit_index=u['unit_index'],A_global=r['A_global'],local_std=r['local_std'],video_ids='1482;1509;1481;2040;2014;2016;2738;2734;2726',labels='39;39;39;53;53;53;72;72;72',checkpoint='/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt',temporal_positions='0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15'))
    write(out/'task037_temporal_gap_masking_manifest.csv',manifest)
    mot=[r for r in selected if r['cohort']=='motivation']; ctrl=[r for r in selected if r['cohort']=='consistent_control']
    write(out/'task037_temporal_gap_motivation_pairs.csv',mot); write(out/'task037_temporal_gap_control_pairs.csv',ctrl)
    # Copy the diagnostic runner into this output's code provenance.
    code=Path(__file__).with_name('task037_temporal_redundancy_motivation.py')
    shutil.copy2(code,out/'task037_temporal_redundancy_motivation_runner.py')
    s=dict(stage='type_stratified_selection',categories=['FF','AA','AF'],selected_pairs=len(selected),selected_pair_ids=[r['pair_id'] for r in selected],decision='TEMPORAL_FUNCTIONAL_SUBSTITUTABILITY_GAP_PARTIALLY_SUPPORTED_PENDING_MASKING',source_output=str(src))
    (out/'task037_temporal_substitutability_type_selection.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    print(json.dumps(s))

if __name__=='__main__': main()
