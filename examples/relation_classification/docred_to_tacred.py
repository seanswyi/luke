import argparse
from itertools import permutations
import json
import os

from tqdm import tqdm


def adjust_mention_positions(entities, text):
    for entity_idx, entity in enumerate(entities):
        entity = sorted(entity, key=lambda x: x['sent_id'])
        for mention_idx, mention in enumerate(entity):
            sent_id = mention['sent_id']

            # No need to adjust span for first sentence.
            if sent_id == 0:
                continue

            prev_sent_lengths = sum([len(x) for x in text[:sent_id]])
            adjusted_positions = [i + prev_sent_lengths for i in mention['pos']]
            entity[mention_idx]['pos'] = adjusted_positions

    return entities


def process(data_file):
    """
    Function to convert DocRED to TACRED format.

    DocRED keys: vertexSet, labels, title, sents
    TACRED keys: id, docid, relation, token, subj_start, subj_end, obj_start, obj_end \
                   subj_type, obj_type, stanford_pos, stanford_ner, stanford_head, \
                   stanford_deprel
    """
    with open(file=data_file) as f:
        data = json.load(fp=f)

    converted_data = []

    progress_bar = tqdm(iterable=data, desc=f"Processing {data_file}", total=len(data))
    for sample in progress_bar:
        text = sample['sents']
        entities = sample['vertexSet']
        entities = adjust_mention_positions(entities, text)
        triplets = sample['labels']
        title = sample['title']

        entity_idxs = list(range(len(entities)))
        all_entity_pairs = list(permutations(iterable=entity_idxs, r=2))

        positive_pairs = [(x['h'], x['t']) for x in triplets]
        negative_pairs = [pair for pair in all_entity_pairs if pair not in positive_pairs]

        # Positive sample loop.
        for triplet in triplets:
            template = {'docid': '',
                        'relation': '',
                        'token': [],
                        'subj_start': [],
                        'subj_end': [],
                        'obj_start': [],
                        'obj_end': [],
                        'subj_type': '',
                        'obj_type': ''}

            head = triplet['h']
            tail = triplet['t']
            relation = triplet['r']

            head_entity_mentions = entities[head]
            tail_entity_mentions = entities[tail]

            subj_starts = [x['pos'][0] for x in head_entity_mentions]
            subj_ends = [x['pos'][1] for x in head_entity_mentions]
            obj_starts = [x['pos'][0] for x in tail_entity_mentions]
            obj_ends = [x['pos'][1] for x in tail_entity_mentions]

            subj_type = head_entity_mentions[0]['type']
            obj_type = tail_entity_mentions[0]['type']

            template['docid'] = title
            template['relation'] = relation
            template['token'] = sum(text, [])
            template['subj_start'] = subj_starts
            template['subj_end'] = subj_ends
            template['obj_start'] = obj_starts
            template['obj_end'] = obj_ends
            template['subj_type'] = subj_type
            template['obj_type'] = obj_type

            converted_data.append(template)

        for negative_pair in negative_pairs:
            template = {'docid': '',
                        'relation': '',
                        'token': [],
                        'subj_start': [],
                        'subj_end': [],
                        'obj_start': [],
                        'obj_end': [],
                        'subj_type': '',
                        'obj_type': ''}

            head = negative_pair[0]
            tail = negative_pair[1]
            relation = 'Na'

            head_entity_mentions = adjust_positions(mentions=entities[head], text=text)
            tail_entity_mentions = adjust_positions(mentions=entities[tail], text=text)

            subj_starts = [x['pos'][0] for x in head_entity_mentions]
            subj_ends = [x['pos'][1] for x in head_entity_mentions]
            obj_starts = [x['pos'][0] for x in tail_entity_mentions]
            obj_ends = [x['pos'][1] for x in tail_entity_mentions]

            subj_type = head_entity_mentions[0]['type']
            obj_type = tail_entity_mentions[0]['type']

            template['docid'] = title
            template['relation'] = relation
            template['token'] = sum(text, [])
            template['subj_start'] = subj_starts
            template['subj_end'] = subj_ends
            template['obj_start'] = obj_starts
            template['obj_end'] = obj_ends
            template['subj_type'] = subj_type
            template['obj_type'] = obj_type

            converted_data.append(template)

    return converted_data


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    train_file = os.path.join(args.docred_dir, args.train_file)
    dev_file = os.path.join(args.docred_dir, args.dev_file)
    test_file = os.path.join(args.docred_dir, args.test_file)

    converted_train_data = process(train_file)
    converted_dev_data = process(dev_file)

    train_savefile = os.path.join(args.save_dir, 'train_tacred.json')
    dev_savefile = os.path.join(args.save_dir, 'dev_tacred.json')

    with open(file=train_savefile, mode='w') as f:
        json.dump(obj=converted_train_data, fp=f, indent=2)

    with open(file=dev_savefile, mode='w') as f:
        json.dump(obj=converted_dev_data, fp=f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--docred_dir', default='/hdd1/seokwon/data/DocRED/original', type=str)
    parser.add_argument('--train_file', default='train_annotated.json', type=str)
    parser.add_argument('--dev_file', default='dev.json', type=str)
    parser.add_argument('--test_file', default='test.json', type=str)
    parser.add_argument('--save_dir', default='/hdd1/seokwon/data/DocRED/TACRED-style', type=str)

    args = parser.parse_args()

    main(args)
