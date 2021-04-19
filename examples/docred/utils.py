from itertools import permutations
import json
import os

import numpy as np
from tqdm import tqdm
from transformers.tokenization_roberta import RobertaTokenizer


with open(file='/hdd1/seokwon/data/DocRED/rel_info.json') as f:
    docred_id2rel = json.load(fp=f)


with open(file='/hdd1/seokwon/data/DocRED/DocRED_baseline_metadata/rel2id.json') as f:
    docred_rel2id = json.load(fp=f)


HEAD_TOKEN = '[HEAD]'
TAIL_TOKEN = '[TAIL]'
ENTITY_MARKER = '[ENTITY]'


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


class InputExample(object):
    def __init__(self, id_, text, entity_pos, head_tail_idxs, labels):
        self.id = id_
        self.text = text
        self.entity_pos = entity_pos
        self.head_tail_idxs = head_tail_idxs
        self.labels = labels


class InputFeatures(object):
    def __init__(
        self,
        title,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        label,
        head_tail_idxs
    ):
        self.title = title
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        self.label = label
        self.head_tail_idxs = head_tail_idxs


class DocumentDatasetProcessor(object):
    def get_train_examples(self, data_dir, debug=False):
        return self._create_examples(data_dir=data_dir, set_type='train', debug=debug)

    def get_dev_examples(self, data_dir, debug=False):
        return self._create_examples(data_dir=data_dir, set_type='dev', debug=debug)

    def get_test_examples(self, data_dir, debug=False):
        return self._create_examples(data_dir=data_dir, set_type='test', debug=debug)

    def get_label_list(self, data_dir, examples=None, debug=False):
        if examples == None:
            examples = self.get_train_examples(data_dir, debug=debug)

        pbar = tqdm(iterable=examples, desc="Getting labels", total=len(examples))

        labels = set()
        for example in pbar:
            labels.add(example.label)

        labels.discard('Na')

        return ['Na'] + sorted(labels)

    def _create_examples(self, data_dir, set_type, debug=False):
        filename = os.path.join(data_dir, set_type + '.json')

        with open(file=filename) as f:
            data = json.load(f)

        if debug:
            data = data[:500]

        examples = []
        pbar = tqdm(iterable=data, desc=f"Processing document data for {set_type}", total=len(data))
        for idx, item in enumerate(pbar):
            words = item['token']
            subj_starts = item['subj_start']
            subj_ends = item['subj_end']
            obj_starts = item['obj_start']
            obj_ends = item['obj_end']
            title = item['docid']

            if item['relation'] == 'Na':
                relation = 'no_relation'
            elif item['relation'] != 'Na':
                relation = docred_id2rel[item['relation']]

            subj_spans = list(zip(subj_starts, subj_ends))
            obj_spans = list(zip(obj_starts, obj_ends))

            mention_spans = dict(subj=subj_spans, obj=obj_spans)

            text = ''
            current_idx = 0
            word2char_map = {}

            for word_idx, word in enumerate(words):
                word2char_map[word_idx] = list(range(current_idx, current_idx + len(word) + 1))
                current_idx = word2char_map[word_idx][-1] + 1

            char_spans = dict(subj=[], obj=[])
            for entity in mention_spans:
                for mention_span in mention_spans[entity]:
                    word_lvl_start = mention_span[0]
                    word_lvl_end = mention_span[1]

                    char_lvl_start = word2char_map[word_lvl_start][0]
                    char_lvl_end = word2char_map[word_lvl_end - 1][-1]

                    char_spans[entity].append((char_lvl_start, char_lvl_end))

            input_example = InputExample(f'{set_type}-{idx}-{title}',
                                         ' '.join(words),
                                         char_spans['subj'],
                                         char_spans['obj'],
                                         item['subj_type'],
                                         item['obj_type'],
                                         item['relation'])
            examples.append(input_example)

        return examples


class DatasetProcessor(object):
    def get_train_examples(self, data_dir, debug=False):
        return self._create_examples(data_dir, 'train')

    def get_dev_examples(self, data_dir, debug=False):
        return self._create_examples(data_dir, 'dev')

    def get_test_examples(self, data_dir, debug=False):
        return self._create_examples(data_dir, 'test')

    def get_label_list(self, data_dir, examples=None):
        labels = set()

        for example in examples:
            labels_ = example.labels

            for label in labels_:
                for idx, element in enumerate(label):
                    if element != 0:
                        labels.add(idx)

        return sorted(labels)

    def _create_examples(self, data_dir, set_type, debug=False):
        """
        Creates individual samples and stores them in InputExample objects.
        """
        if set_type == 'train':
            set_type = 'train_annotated'

        with open(file=os.path.join(data_dir, set_type + '.json')) as f:
            data = json.load(f)

        examples = []
        pbar = tqdm(iterable=data, desc=f"Processing {set_type}", total=len(data))
        for i, item in enumerate(pbar):
            sentences = item['sents']
            triplets = item['labels']
            entities = adjust_mention_positions(entities=item['vertexSet'], text=sentences)
            title = item['title']

            entity_pos = []
            for entity in entities:
                mention_positions = []
                for mention in entity:
                    mention_positions.append(mention['pos'])

                entity_pos.append(mention_positions)

            head_tail_pair2relation = {}
            for triplet in triplets:
                relation = int(docred_rel2id[triplet['r']])
                head_idx = triplet['h']
                tail_idx = triplet['t']
                head_tail_pair = (head_idx, tail_idx)

                if head_tail_pair in head_tail_pair2relation:
                    head_tail_pair2relation[head_tail_pair].append(relation)
                else:
                    head_tail_pair2relation[head_tail_pair] = [relation]

            whole_text = ' '.join([' '.join(sentence) for sentence in sentences]).split()
            entity_idxs = list(range(len(entities)))
            all_entity_pairs = list(permutations(iterable=entity_idxs, r=2))

            positive_pairs = list(head_tail_pair2relation.keys())
            negative_pairs = [pair for pair in all_entity_pairs if pair not in positive_pairs]

            relations = []
            head_tail_pairs = []
            for head_tail_pair in head_tail_pair2relation:
                relation = [0] * len(docred_rel2id)

                for sample in head_tail_pair2relation[head_tail_pair]:
                    relation[sample] = 1

                relations.append(relation)
                head_tail_pairs.append(head_tail_pair)

            for head_idx, _ in enumerate(entities):
                for tail_idx, _ in enumerate(entities):
                    head_tail_pair = (head_idx, tail_idx)

                    if (head_idx != tail_idx) and (head_tail_pair not in head_tail_pairs):
                        relation = [1] + ([0] * (len(docred_rel2id) - 1))
                        relations.append(relation)
                        head_tail_pairs.append(head_tail_pair)

            example = InputExample(f'{title}', # Need to change this to be title.
                                   ' '.join(whole_text),
                                   entity_pos,
                                   head_tail_pairs,
                                   relations)
            examples.append(example)

        return examples


def convert_examples_to_features(examples, label_list, tokenizer, max_mention_length, max_seq_length):
    label_map = {l: i for i, l in enumerate(label_list)}

    def tokenize(text):
        if isinstance(tokenizer, RobertaTokenizer):
            return tokenizer.tokenize(text, add_prefix_space=True)
        else:
            return tokenizer.tokenize(text)

    features = []
    pbar = tqdm(iterable=examples, desc="Converting examples to features", total=len(examples))
    for example in pbar:
        text = example.text
        entity_positions = example.entity_pos
        labels = example.labels
        head_tail_idxs = example.head_tail_idxs
        title = example.id.split('-')[0]

        # Ensuring mention positions are in ascending order is crucial, otherwise the spans will get messed up.
        for idx, entity in enumerate(entity_positions):
            entity = sorted(entity, key=lambda x: x[0])
            entity_positions[idx] = entity

        # Get character-level spans for entity mentions. This is used for inserting the entity markers.
        words = text.split()
        text_ = ''
        current_idx = 0
        char_lvl_spans = []
        for entity in entity_positions:
            text_ = ''
            current_idx = 0
            mention_char_spans = []

            for mention_span in entity:
                text_ += ' '.join(words[current_idx:mention_span[0]])

                if text_:
                    text_ += ' '

                start = len(text_)
                text_ += ' '.join(words[mention_span[0]:mention_span[1]]) + ' '

                end = len(text_)
                current_idx = mention_span[1]

                mention_char_spans.append([start, end])

            char_lvl_spans.append(mention_char_spans)
        ###########################################################################################

        # Insert entity markers in the front and back of entity mentions. #########################
        text_sliced_entities = []
        all_spans = sorted(sum(char_lvl_spans, []), key=lambda x: x[0])
        current_idx = 0
        for span in all_spans:
            start = span[0]
            end = span[1]

            text_slice = text[current_idx:start]
            text_sliced_entities.append(text_slice)
            text_sliced_entities.append(text[start:end])
            current_idx = end

        text_sliced_entities.append(text[current_idx:])
        for idx, text_slice in enumerate(text_sliced_entities):
            text_sliced_entities[idx] = text_slice.strip()

        entity_marked_text = (' ' + ENTITY_MARKER + ' ').join(text_sliced_entities)
        entity_marked_text = ' '.join(entity_marked_text.split())
        ###########################################################################################

        # Mark which entity each span belongs to. #################################################
        for idx, entity in enumerate(char_lvl_spans):
            marked_mentions = []
            for mention in entity:
                entity_id_marked = (idx, mention)
                marked_mentions.append(entity_id_marked)

            char_lvl_spans[idx] = marked_mentions
        ###########################################################################################

        # Adjust spans to account for entity markers. #############################################
        all_spans = sorted(sum(char_lvl_spans, []), key=lambda x: x[1][0])
        current_idx = 0
        adjustment_length = (len(ENTITY_MARKER) * 2) + 2 # Two markers with two spaces.
        for idx, id_span in enumerate(all_spans):
            entity_id = id_span[0]
            span = id_span[1]

            original_start = span[0]
            original_end = span[1]

            new_start = original_start + (adjustment_length * current_idx)
            new_end = original_end + (adjustment_length * (current_idx + 1))
            new_span = [new_start, new_end]

            new_pair = (entity_id, new_span)
            all_spans[idx] = new_pair

            current_idx += 1
        ###########################################################################################

        # Get token-level spans. ##################################################################
        tokens = [tokenizer.cls_token]
        current_idx = 0
        token_lvl_spans = []
        for idx, id_span in enumerate(all_spans):
            entity_id = id_span[0]
            span = id_span[1]

            start = span[0]
            end = span[1]

            text_chunk = entity_marked_text[current_idx:start]
            tokens += tokenize(text_chunk)

            token_start = len(tokens)

            entity_text_chunk = entity_marked_text[start:end]
            tokens += tokenize(entity_text_chunk)

            token_end = len(tokens)

            token_lvl_id_span = (entity_id, [token_start, token_end])
            token_lvl_spans.append(token_lvl_id_span)

            current_idx = end

        tokens += tokenize(entity_marked_text[current_idx:]) + [tokenizer.sep_token]
        ###########################################################################################

        word_ids = tokenizer.convert_tokens_to_ids(tokens)
        # word_ids = word_ids[:max_seq_length]

        word_attention_mask = [1] * len(tokens)
        # word_attention_mask = word_attention_mask[:max_seq_length]

        word_segment_ids = [0] * len(tokens)
        # word_segment_ids = word_segment_ids[:max_seq_length]

        entity_ids = [1, 2]
        entity_segment_ids = [0, 0]
        entity_attention_mask = [1, 1]

        num_entities = len(entity_positions)
        entity_position_ids = {idx: [] for idx in range(num_entities)}
        for id_span in token_lvl_spans:
            entity_id = id_span[0]
            span = id_span[1]
            entity_position_ids[entity_id].append(span)

        entity_position_ids = list(entity_position_ids.values())

        luke_style_position_ids = []
        for entity in entity_position_ids:
            entity_full_idxs = []
            for mention in entity:
                mention_full_idxs = [-1] * max_mention_length
                mention_idxs = list(range(mention[0], mention[1]))
                mention_length = len(mention_idxs)
                mention_full_idxs[:mention_length] = mention_idxs
                entity_full_idxs.append(mention_full_idxs)
            luke_style_position_ids.append(entity_full_idxs)

        labels_ = []
        for label in labels:
            labels_.append(np.argmax(label))

        feature = InputFeatures(title=title,
                                word_ids=word_ids,
                                word_segment_ids=word_segment_ids,
                                word_attention_mask=word_attention_mask,
                                entity_ids=entity_ids,
                                entity_position_ids=luke_style_position_ids,
                                entity_segment_ids=entity_segment_ids,
                                entity_attention_mask=entity_attention_mask,
                                label=labels_,
                                head_tail_idxs=head_tail_idxs)
        features.append(feature)

    return features
