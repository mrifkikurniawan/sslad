# naive fine-tuning
nohup python3 classification.py \
--name result/ \
--root logs/ \
--num_workers 4 \
--store \
--gpu_id 4 \
--config configs/naive_finetune.yaml \
> nohup/naive-finetune.out &