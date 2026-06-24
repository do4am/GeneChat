CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port=25702 train_esm.py  \
	--cfg-path configs/genechat_stage1.yaml \
	--options run.resume_ckpt_path=/home/namdo/applications/GeneChat/exon_count/data/checkpoints/20251216095/checkpoint_35000.pth run.checkpoint_freq=2 run.max_checkpoints=3
