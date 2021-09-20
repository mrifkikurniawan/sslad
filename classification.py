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


def main():
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

    ######################################
    #                                    #
    # Editing below this line allowed    #
    #                                    #
    ######################################
    seed = 0
    args.root = f"{args.root}/SSLAD-2D/labeled"
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
    for k in config.method.args.keys():
        print(f"{k}: {config.method.args[k]}")
        
    # logging
    hparams = edict(method=config.method)
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
    train_sets = create_train_set(args.root, img_size)
    evaluate = 'test' if args.test else 'val'
    if evaluate == "val":
        test_sets = create_val_set(args.root, img_size)
    else:
        test_sets, _ = create_test_set_from_pkl(args.root, img_size)

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
    
if __name__ == '__main__':
    main()
