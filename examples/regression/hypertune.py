from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
import pickle
import uuid
import json
from datetime import datetime
NOW = datetime.now()
from tqdm import tqdm, trange

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForRegression
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from helpers import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

space = {
    # This loguniform scale will multiply the learning rate, so as to make
    # it vary exponentially, in a multiplicative fashion rather than in
    # a linear fashion, to handle his exponentialy varying nature:
    'lr_rate_mult': hp.loguniform('lr_rate_mult', -1, 1),
    # Batch size fed for each gradient update
    'batch_size': hp.quniform('batch_size', 8, 20, 4),
    # Choice of optimizer:
    # 'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
    # Uniform distribution in finding appropriate dropout values, FC layers
    ##'fc_dropout_drop_prob': hp.uniform('fc_dropout_prob', 0.0, 0.6),
    # Number of hidden layers in the Transformer encoder
    ##'num_hidden_layers': hp.quniform('num_hidden_layers', 8, 14, 1)
    # Use batch normalisation at more places?
    #'use_BN': hp.choice('use_BN', [False, True]),
    # Let's multiply the "default" number of hidden units:
    'fc_hiddn_units': hp.quniform('fc_hiddn_units', 50, 400, 50),
    # Use one more FC layer at output
    'one_more_fc': hp.choice(
        'one_more_fc', [None, hp.quniform('fc_2_units', 50, 400, 50),]
    ),
    # Activations that are used everywhere
    ##'activation': hp.choice('activation', ['relu', 'gelu', 'swish'])
}
    

def build_model(hype_space, args, device):
    print("Current space being optimized:")
    print(hype_space)

    # Prepare model
    model = BertForRegression.from_pretrained(args.bert_model, inner_layer_size=int(hype_space['fc_hiddn_units']),
              outer_layer_size=hype_space['one_more_fc'],
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank)) 
              # max_position_embeddings=256, hidden_dropout_prob=hype_space['fc_dropout_prob'], 
              # num_hidden_layers=hype_space['num_hidden_layers'], hidden_act=hype_space['activation'])
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model

def build_and_optimize(hype_space, args, device, train_data, train_sampler, eval_data, eval_sampler):
    """Build a convolutional neural network and train it."""
    
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=int(hype_space['batch_size']))
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=int(hype_space['batch_size']))
    model = build_model(hype_space, args, device)

    global_step = 0
    t_total = int(len(train_data) / hype_space['batch_size'] * args.num_train_epochs)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        
    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate*hype_space['lr_rate_mult'],
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    ##TODO make this cleaner
    n_gpu = torch.cuda.device_count() 
    model.train()
    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, scores = batch
            loss = model(input_ids, segment_ids, input_mask, scores.float())
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            
            # modify learning rate with special warm up BERT uses
            lr = args.learning_rate*hype_space['lr_rate_mult']
            lr_this_step =  lr* warmup_linear(global_step/t_total, args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        logger.info("Loss at epoch %d: %f",  e, tr_loss)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, scores in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        scores = scores.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, scores.float())
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        scores = scores.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, scores)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    results = {'eval_loss': eval_loss,
              'eval_accuracy': int(eval_accuracy),
              'global_step': global_step,
              'loss': -int(eval_accuracy),
              'status': STATUS_OK}

    print("RESULTS:")
    print(json.dumps(
        results,
        sort_keys=True,
        indent=4, separators=(',', ': ')
    ))
    # Save all training results to disks with unique filenames
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(args.output_dir + 'model_{}_{}_{}.json'.format(str(eval_accuracy), str(uuid.uuid1())[:3], NOW.strftime("%m_%d_%H_%M")), 'w') as f:
        json.dump(
            hype_space, f,
            sort_keys=True,
            indent=4, separators=(',', ': ')
        )
        json.dump(
            results, f,
            sort_keys=True,
            indent=4, separators=(',', ': ')
        )
    #clear up Cuda space FOR SURE. repository issue as of the writing of this line
    del model

    return results
    print("\n\n")

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--num_train_epochs",
                        default=5.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.") 
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()
    processors = {
        "sts": StsProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_examples = processor.get_train_examples(args.data_dir)
    
    train_features = convert_examples_to_features(
            train_examples, args.max_seq_length, tokenizer, logger=logger)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_scores = torch.tensor([f.score for f in train_features], dtype=torch.float)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_scores)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_scores = torch.tensor([f.score for f in eval_features], dtype=torch.float)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_scores)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    

    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 10

    print("Attempt to resume a past training if it exists:")

    try:
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        lambda x: build_and_optimize(x, args, device, train_data, train_sampler, eval_data, eval_sampler),
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open("results.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")
    print("Best results yet (note that this is NOT calculated on the 'loss' "
          "metric despite the key is 'loss' - we rather take the negative "
          "best accuracy throughout learning as a metric to minimize):")
    print(best) 


if __name__ == "__main__":
    main()