export CUDA_VISIBLE_DEVICES=1
set -euo pipefail
export PYTHONPATH=/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy/src/lgfr_runtime:/home/jixinye25/jxy_work1/swintrans_task037_complete/verification:${PYTHONPATH:-}
/home/jixinye25/miniconda3/envs/MC_Pruning/bin/python -u /home/jixinye25/jxy_work1/swintrans_task037_complete/verification/task037_context_proxy_gpu.py \
 --project_root /home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy \
 --checkpoint /home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt \
 --output_dir /data/jixinye25/work1/output/task037_context_proxy/shards/gpu1 \
 --device cuda:0 \
 --adapter ucf101_videoswin_probe_adapter_v2 \
 --val_list /data/jixinye25/work1/output/task037_bms_confirm30/task_bms_confirm30_gpu1_val_list.txt \
 --context_manifest /data/jixinye25/work1/output/task037_bms_confirm30/task_bms_confirm30_context_manifest.csv \
 --confirmation_output /data/jixinye25/work1/output/task037_bms_confirm30 \
 --unit_manifest /data/jixinye25/work1/output/task037_bms_temporal_validation/task_bms_temporal_unit_manifest.csv \
 --domain_manifest /data/jixinye25/work1/output/task037_bms_temporal_validation/task_bms_temporal_domain_manifest.csv \
 --video_manifest /data/jixinye25/work1/output/task037_bms_confirm30/task_bms_confirm30_video_manifest.csv \
 --frame_root /data/jixinye25/UCF101_Frame/frames \
 --num_classes 5 --videos_per_class 3 --num_workers 2 --seed 3407

