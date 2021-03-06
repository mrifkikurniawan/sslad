import torch

import argparse
import yaml
from easydict import EasyDict as edict

from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, class_accuracy_metrics
from avalanche.logging import TextLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive

from class_strategy import *
from classification_util import *
from utils import create_instance, seed_everything
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.trial import TrialState

def main():
    global args, hparams_optimizer_cfg
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='result',
                        help='Name of the result files')
    parser.add_argument('--root', default="../data",
                        help='Root folder where the data is stored')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers to use for dataloading. Recommended to have more than 1')
    parser.add_argument('--store', action='store_true',
                        help="If set the prediciton files required for submission will be created")
    parser.add_argument('--test', action='store_true',
                        help='If set model will be evaluated on test set, else on validation set')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If set, training will be on the CPU')
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='training GPU id')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='path to training/method yaml configuration file')
    parser.add_argument('--comment', type=str, default='',help='comment to tensorboard logger')
    parser.add_argument('--store_model', action='store_true',
                        help="Stores model if specified. Has no effect is store is not set")
    args = parser.parse_args()
    config = edict(yaml.safe_load(open(args.config, "r")))
    hparams_optimizer_cfg = config.hparams_optimizer
    
    ######################################
    #                                    #
    # Editing below this line allowed    #
    #                                    #
    ######################################
    
def train(trial: optuna.trial.Trial):
    seed = 0
    root = f"{args.root}/SSLAD-2D/labeled"
    config = edict(yaml.safe_load(open(args.config, "r")))
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")
    logger = SummaryWriter(log_dir=args.name, comment=args.comment)
    seed_everything(seed)

    # print configuration
    print("--------Configuration--------")
    print(f"gpu_id: {args.gpu_id}")
    print(f"log path: {args.name}")
    print(f"num workers: {args.num_workers}")
    print(f"store prediction: {args.store}")
    print(f"eval on test set: {args.test}")
    print(f"method: {config.method.method}")
        
    # logging
    hparams = edict(method=config.method)
    
    # --------- Optuna Hparams Optimizer ---------
    # set hparams optimizer val to method config
    for hparam in hparams_optimizer_cfg.hparams:
        suggest_method = getattr(trial, hparam.method)
        hparam_val = suggest_method(**hparam.trial_args)
        param_name = hparam.name.split('.')
        temporary_cfg = config.method.args
          
        # set proposed hparam value to config file 
        for i, attr in enumerate(param_name):
            if isinstance(temporary_cfg, list):
                temporary_cfg = temporary_cfg[int(hparam.index)]
            if i < len(param_name) - 1:
                temporary_cfg = temporary_cfg.get(attr)
            elif i == len(param_name)-1:
                temporary_cfg[attr] = hparam_val
    
    # print hparams
    for k in config.method.args.keys():
        print(f"{k}: {config.method.args[k]}")
    
    for k in hparams.keys():
        logger.add_text(k, str(hparams[k]))

    method = create_instance(config.method)
    model = method.model

    optimizer = method.optimizer
    criterion = method.criterion
    batch_size = 10

    # Add any additional plugins to be used by Avalanche to this list. A template
    # is provided in class_strategy.py.
    plugins = [method.plugins]
    logger_ext = method.logger

    ######################################
    #                                    #
    # No editing below this line allowed #
    #                                    #
    ######################################

    if batch_size > 10:
       raise ValueError(f"Batch size {batch_size} not allowed, should be less than or equal to 10")

    img_size = 64
    train_sets = create_train_set(root, img_size)
    evaluate = 'test' if args.test else 'val'
    if evaluate == "val":
        test_sets = create_val_set(root, img_size)
    else:
        test_sets, _ = create_test_set_from_pkl(root, img_size)

    benchmark = create_multi_dataset_generic_benchmark(train_datasets=train_sets, test_datasets=test_sets)

    text_logger = TextLogger(open(f"./{args.name}.log", 'w'))
    interactive_logger = InteractiveLogger()
    store = args.name if args.store else None

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True), loss_metrics(stream=True),
        class_accuracy_metrics(stream=True),
        ClassEvaluationPlugin(reset_at='stream', emit_at='stream', mode='eval',
                              store=store),
        loggers=[text_logger, interactive_logger])

    strategy = Naive(
        model, optimizer, criterion, train_mb_size=batch_size, train_epochs=1, eval_mb_size=256, device=device,
        evaluator=eval_plugin, eval_every=1, plugins=plugins)

    accuracies_test = []
    for i, experience in enumerate(benchmark.train_stream):
        # Shuffle will be passed through to dataloader creator.
        strategy.train(experience, eval_streams=[], shuffle=False, num_workers=args.num_workers)

        results = strategy.eval(benchmark.test_stream, num_workers=args.num_workers)
        mean_acc = [r[1] for r in results['Top1_ClassAcc_Stream/eval_phase/test_stream/Task000']]
        accuracies_test.append(sum(mean_acc) / len(mean_acc))
        logger_ext.log({"test/accuracy": sum(mean_acc) / len(mean_acc)})

    print(f"Average mean test accuracy: {sum(accuracies_test) / len(accuracies_test) * 100:.3f}%")
    print(f"Average mean test accuracy: {sum(accuracies_test) / len(accuracies_test) * 100:.3f}%",
          file=open(f'./{args.name}.log', 'a'))
    print(f"Final mean test accuracy: {accuracies_test[-1] * 100:.3f}%")
    print(f"Final mean test accuracy: {accuracies_test[-1] * 100:.3f}%",
          file=open(f'./{args.name}.log', 'a'))

    logger.add_scalar("Average mean test accuracy", sum(accuracies_test) / len(accuracies_test) * 100)
    logger.add_scalar("Final mean test accuracy", accuracies_test[-1] * 100)
    
    logger_ext.log({'test/final mean acc': sum(accuracies_test) / len(accuracies_test) * 100,
                'test/average mean acc': accuracies_test[-1]
                })
    
    if args.store_model:
        torch.save(model.state_dict(), f'./{args.name}.pt')
    
    # finish wandb logger
    logger_ext.finish()
    
    return sum(accuracies_test) / len(accuracies_test) * 100
    
if __name__ == '__main__':  
    main()
    
    # optuna hparams optimizer
    study = optuna.create_study(direction=hparams_optimizer_cfg.direction)
    study.optimize(train, n_trials=hparams_optimizer_cfg.n_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("----- Study statistics -----")
    print("Number of finished trials: ", len(study.trials))
    print("Number of pruned trials: ", len(pruned_trials))
    print("Number of complete trials: ", len(complete_trials))

    # best trial
    print(f"Best trial: {study.best_trial}")
    print(f"Best params: {study.best_params}")
    print(f"Best metric acc: {study.best_value}")