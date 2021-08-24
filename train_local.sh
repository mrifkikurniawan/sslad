python3 classification.py \
--name logs/finetune_iid/naive_finetune_iid \
--root ../../datasets \
--num_workers 8 \
--store \
--gpu_id 0 \
--config configs/naive_finetune.yaml \
--comment naive_finetune_iid \
