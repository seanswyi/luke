import json
import logging
import os
from argparse import Namespace
import pickle

import click
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import WEIGHTS_NAME
import wandb

from luke.utils.entity_vocab import MASK_TOKEN

from ..utils import set_seed
from ..utils.trainer import Trainer, trainer_args
from .model import LukeForRelationClassification
from .utils import HEAD_TOKEN, TAIL_TOKEN, convert_examples_to_features, DatasetProcessor, DocumentDatasetProcessor


logger = logging.getLogger(__name__)


@click.group(name="relation-classification")
def cli():
    pass


@cli.command()
@click.option("--checkpoint-file", type=click.Path(exists=True))
@click.option("--data-dir", default="data/tacred", type=click.Path(exists=True))
@click.option("--do-eval/--no-eval", default=True)
@click.option("--do-train/--no-train", default=True)
@click.option("--eval-batch-size", default=128)
@click.option("--num-train-epochs", default=5.0)
@click.option("--seed", default=42)
@click.option("--train-batch-size", default=4)

@click.option('--setting', default='sentence')
@click.option('--debug/--no-debug', default=False)
@click.option('--multi-gpu/--no-multi-gpu', default=False)
@click.option('--atloss/--no-atloss', default=False)

@click.option('--classifier', default='linear')

@trainer_args
@click.pass_obj
def run(common_args, **task_args):
    logger.info("Started process.")
    task_args.update(common_args)
    args = Namespace(**task_args)

    if args.atloss:
        wandb_name = f'TACRED_{args.classifier}_{args.learning_rate}_atloss'
    elif not args.atloss:
        wandb_name = f'TACRED_{args.classifier}_{args.learning_rate}'

    wandb.init(project="LUKE DocRED", name=wandb_name)

    set_seed(args.seed)

    args.experiment.log_parameters({p.name: getattr(args, p.name) for p in run.params})

    logger.info("Starting embedding and tokenizer configuration.")
    args.model_config.vocab_size += 2
    word_emb = args.model_weights["embeddings.word_embeddings.weight"]
    head_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
    tail_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)
    args.model_weights["embeddings.word_embeddings.weight"] = torch.cat([word_emb, head_emb, tail_emb])
    args.tokenizer.add_special_tokens(dict(additional_special_tokens=[HEAD_TOKEN, TAIL_TOKEN]))

    entity_emb = args.model_weights["entity_embeddings.entity_embeddings.weight"]
    mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0).expand(2, -1)
    args.model_config.entity_vocab_size = 3
    args.model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb[:1], mask_emb])

    train_dataloader, _, _, label_list = load_examples(args, fold='train', setting=args.setting)
    num_labels = len(label_list)

    results = {}
    if args.do_train:
        model = LukeForRelationClassification(args, num_labels)
        model.load_state_dict(args.model_weights, strict=False)

        if args.multi_gpu:
            model = torch.nn.DataParallel(model)

        model.to(args.device)

        num_train_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

        best_dev_f1 = [-1]
        best_weights = [None]

        def step_callback(model, global_step):
            if global_step % num_train_steps_per_epoch == 0 and args.local_rank in (0, -1):
                epoch = int(global_step / num_train_steps_per_epoch - 1)
                dev_results = evaluate(args, model, fold="dev")

                wandb.log(dev_results)

                args.experiment.log_metrics({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()}, epoch=epoch)
                results.update({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()})
                tqdm.write("dev: " + str(dev_results))

                if dev_results["f1"] > best_dev_f1[0]:
                    if hasattr(model, "module"):
                        best_weights[0] = {k: v.to("cpu").clone() for k, v in model.module.state_dict().items()}
                    else:
                        best_weights[0] = {k: v.to("cpu").clone() for k, v in model.state_dict().items()}
                    best_dev_f1[0] = dev_results["f1"]
                    results["best_epoch"] = epoch

                model.train()

        trainer = Trainer(
            args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps, step_callback=step_callback
        )
        trainer.train()

    if args.do_train and args.local_rank in (0, -1):
        logger.info("Saving the model checkpoint to %s", args.output_dir)
        torch.save(best_weights[0], os.path.join(args.output_dir, WEIGHTS_NAME))

    if args.local_rank not in (0, -1):
        return {}

    model = None
    torch.cuda.empty_cache()

    if args.do_eval:
        model = LukeForRelationClassification(args, num_labels)
        if args.checkpoint_file:
            model.load_state_dict(torch.load(args.checkpoint_file, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location="cpu"))
        model.to(args.device)

        if args.setting == 'sentence':
            eval_sets = ['dev', 'test']
        elif args.setting == 'document':
            eval_sets = ['dev']

        for eval_set in eval_sets:
            output_file = os.path.join(args.output_dir, f'{eval_set}_predictions.txt')
            results.update({f'{eval_set}_{k}': v for k, v in evaluate(args, model, eval_set, output_file).items()})

    logger.info("Results: %s", json.dumps(results, indent=2, sort_keys=True))
    args.experiment.log_metrics(results)

    save_file = os.path.join(args.output_dir, 'results.json')
    with open(file=save_file, mode='w') as f:
        json.dump(obj=results, fp=f, indent=2)

    return results


def evaluate(args, model, fold='dev', output_file=None):
    dataloader, _, _, label_list = load_examples(args, fold=fold, setting=args.setting)
    predictions = []
    labels = []

    model.eval()
    for batch in tqdm(dataloader, desc=fold):
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != 'label'}
        with torch.no_grad():
            logits = model(**inputs)

        predictions.extend(logits.detach().cpu().numpy().argmax(axis=1))
        labels.extend(batch['label'].to('cpu').tolist())

    if output_file:
        with open(file=output_file, mode='w') as f:
            for prediction in predictions:
                f.write(label_list[prediction] + '\n')

    num_predicted_labels = 0
    num_gold_labels = 0
    num_correct_labels = 0

    for label, prediction in zip(labels, predictions):
        if prediction != 0:
            num_predicted_labels += 1

        if label != 0:
            num_gold_labels += 1
            if prediction == label:
                num_correct_labels += 1

    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.0

    recall = num_correct_labels / num_gold_labels

    if recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return dict(precision=precision, recall=recall, f1=f1)


def load_examples(args, fold='train', setting='sentence'):
    logger.info("Loading data...")
    if args.local_rank not in (-1, 0) and fold == 'train':
        torch.distributed.barrier()

    if setting == 'sentence':
        processor = DatasetProcessor()
    elif setting == 'document':
        processor = DocumentDatasetProcessor()
        args.data_dir = '/hdd1/seokwon/data/DocRED/TACRED-style'
    else:
        raise NotImplementedError

    if fold == "train":
        examples = processor.get_train_examples(args.data_dir, debug=args.debug)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir, debug=args.debug)
    elif (fold == 'test') and (setting == 'sentence'):
        examples = processor.get_test_examples(args.data_dir, debug=args.debug)

    label_list = processor.get_label_list(args.data_dir, examples=examples)

    if (setting == 'document') and (not args.debug):
        if os.path.exists(f'/hdd1/seokwon/data/DocRED/TACRED-style/features/{fold}_features.pkl'):
            logger.info(f"Loading features from '/hdd1/seokwon/data/DocRED/TACRED-style/features/{fold}_features.pkl'")
            with open(file=f'/hdd1/seokwon/data/DocRED/TACRED-style/features/{fold}_features.pkl', mode='rb') as f:
                features = pickle.load(file=f)
        else:
            logger.info("Creating features from the dataset...")
            features = convert_examples_to_features(examples, label_list, args.tokenizer, args.max_mention_length, setting=setting)
    elif (setting == 'sentence') or args.debug:
        logger.info("Creating features from the dataset...")
        features = convert_examples_to_features(examples, label_list, args.tokenizer, args.max_mention_length, setting=setting)

    # Remove features with word_ids longer than 514. ##############################################
    logger.warning("Filtering out features with text longer than 514.")
    filtered_features = []
    pbar = tqdm(iterable=features, desc=f"Filtering out long features for {setting}-{fold}")
    count = 0
    for feature in pbar:
        if len(feature.word_ids) < 513:
            filtered_features.append(feature)
            count += 1

    logger.info(f"A total of {count} samples remain.")
    features = filtered_features
    ###############################################################################################

    if args.local_rank == 0 and fold == "train":
        torch.distributed.barrier()

    def collate_fn(batch):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o, attr_name), dtype=torch.long) for o in batch]
            padded_tensor = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

            return padded_tensor

        return dict(
            word_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            entity_ids=create_padded_sequence("entity_ids", 0),
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0),
            entity_position_ids=create_padded_sequence("entity_position_ids", -1),
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0),
            label=torch.tensor([o.label for o in batch], dtype=torch.long),
        )

    if fold in ['dev', 'test']:
        dataloader = DataLoader(features, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)

        dataloader = DataLoader(features, sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)

    return dataloader, examples, features, label_list
