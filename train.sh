python3.9 classification.py \
--name logs/cl_strategy/cl_strategy_uncertainty_entropy \
--root /home/rifki/continual_learning/datasets \
--num_workers 32 \
--store \
--gpu_id 0 \
--config configs/cl_strategy.yaml \
--comment cl_strategy_uncertainty_entropy \
--store_model
