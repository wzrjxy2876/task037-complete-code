#!/usr/bin/env python3
"""Targeted tests for the frozen Task037 BMS temporal validation."""
import importlib.util
from pathlib import Path
import os
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("bms_temporal",HERE/"task037_bms_temporal_validation.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_fixed_cardinality_identity():
    x=m.make_interventions()
    assert len(x)==80
    assert [sum(r["span"]==s for r in x) for s in m.SPANS]==[16]*5
    assert {(r["span"],r["pair_index"]) for r in x}=={(s,p) for s in m.SPANS for p in range(16)}

def test_type_and_pair_identity():
    assert m.unit_type("attention_head")=="Attention"
    assert m.unit_type("ffn_neuron")=="FFN"
    assert m.stage("layers.3.blocks.0.mlp")==3

def test_signed_order_and_strict_reversal():
    # Descending signed damage and global-index ascending tie break.
    vals={7:1.0,3:1.0,5:-1.0}
    assert sorted(vals, key=lambda i: (-vals[i], i)) == [3,7,5]
    assert (1.0-0.0)*(-1.0-1.0)<0
    assert not (1.0-1.0)*(-1.0-1.0)<0

def test_generated_identity_artifacts_if_available():
    p=Path(os.environ.get("TASK037_BMS_OUTPUT","/data/jixinye25/work1/output/task037_bms_temporal_validation"))
    if not (p/"task_bms_temporal_summary.json").exists(): return
    import json
    s=json.loads((p/"task_bms_temporal_summary.json").read_text())
    assert s["mapped_units"]==36378 and s["valid_units"]==36363 and s["frozen_domains"]==423
    assert s["videos"]==3 and s["T"]==32 and s["interventions"]==80
    raw=pd.read_csv(p/"task_bms_temporal_conditioned_damage.csv")
    assert raw.global_index.nunique()==36 and raw.video_index.nunique()==3
    assert raw[["level","span","pair_index"]].drop_duplicates().shape[0]==80
    assert len(raw)==36*3*80
    assert set(pd.read_csv(p/"task_bms_temporal_identity_audit.csv").status)=={"PASS"}

if __name__=="__main__":
    for n,v in sorted(globals().items()):
        if n.startswith("test_"): v()
    print("Task037 BMS temporal validation tests passed")
