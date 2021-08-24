python3.9 classification.py \
<<<<<<< HEAD
--name logs/finetune_iid/naive_finetune_iid \
=======
--name logs/replay/replay_class_balanced_random \
>>>>>>> parent of 65b943c... train replay_experienced_balance_random
--root /home/rifki/continual_learning/datasets \
--num_workers 16 \
--store \
--gpu_id 4 \
<<<<<<< HEAD
--config configs/naive_finetune.yaml \
--comment naive_finetune_iid \
--store_model
=======
--config configs/replay/replay_class_balanced_random.yaml \
--comment replay_class_balanced_random
>>>>>>> parent of 65b943c... train replay_experienced_balance_random
