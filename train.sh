python3.9 classification.py \
--name logs/finetune_iid/naive_finetune_iid_bs128 \
--root /home/rifki/continual_learning/datasets \
--num_workers 16 \
--store \
--gpu_id 4 \
--config configs/naive_finetune.yaml \
--comment naive_finetune_iid \
--store_model
