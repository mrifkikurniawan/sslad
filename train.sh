# naive fine-tuning
nohup python3.9 classification.py \
--name finetune \
--root /home/rifki/continual_learning/datasets \
--num_workers 4 \
--store \
--gpu_id 4 \
--config configs/naive_finetune.yaml \
> nohup/naive-finetune.out &