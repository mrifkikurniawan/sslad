# naive fine-tuning
nohup python3.9 classification.py \
--name logs/finetune/finetune_wd_000005 \
--root /home/rifki/continual_learning/datasets \
--num_workers 16 \
--store \
--gpu_id 4 \
--config configs/naive_finetune.yaml \
--comment finetune_wd_000005 \
> nohup/finetune_wd_0001.out &
