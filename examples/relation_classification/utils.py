import json
import os

from tqdm import tqdm
from transformers.tokenization_roberta import RobertaTokenizer


with open(file='/hdd1/seokwon/data/DocRED/rel_info.json') as f:
    docred_id2rel = json.load(fp=f)


HEAD_TOKEN = '[HEAD]'
TAIL_TOKEN = '[TAIL]'


class InputExample(object):
    def __init__(self, id_, text, span_a, span_b, type_a, type_b, label):
        self.id = id_
        self.text = text
        self.span_a = span_a
        self.span_b = span_b
        self.type_a = type_a
        self.type_b = type_b
        self.label = label


class InputFeatures(object):
    def __init__(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        label,
    ):
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        self.label = label


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
        if debug:
            filename = os.path.join(data_dir, set_type + '_debug.json')
        elif not debug:
            filename = os.path.join(data_dir, set_type + '.json')

        with open(file=filename) as f:
            data = json.load(f)

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

    def get_label_list(self, data_dir, debug=False, examples=None):
        labels = set()
        for example in self.get_train_examples(data_dir):
            labels.add(example.label)
        labels.discard('no_relation')
        return ['no_relation'] + sorted(labels)

    def _create_examples(self, data_dir, set_type, debug=False):
        """
        Creates individual samples and stores them in InputExample objects.
        """
        with open(file=os.path.join(data_dir, set_type + '.json')) as f:
            data = json.load(f)

        examples = []
        for i, item in enumerate(data):
            tokens = item["token"]
            token_spans = dict(
                subj=(item["subj_start"], item["subj_end"] + 1), obj=(item["obj_start"], item["obj_end"] + 1)
            )

            if token_spans["subj"][0] < token_spans["obj"][0]:
                entity_order = ("subj", "obj")
            else:
                entity_order = ("obj", "subj")

            # Get character-level spans for entities. #############################################
            text = ''
            cur = 0
            char_spans = dict(subj=[None, None], obj=[None, None])
            for target_entity in entity_order:
                token_span = token_spans[target_entity] # Convert this so that we're looping through all mention spans.

                text += ' '.join(tokens[cur:token_span[0]])

                if text:
                    text += ' '

                char_spans[target_entity][0] = len(text)
                text += ' '.join(tokens[token_span[0]:token_span[1]]) + ' '

                char_spans[target_entity][1] = len(text)
                cur = token_span[1]

            text += ' '.join(tokens[cur:])
            text = text.rstrip()
            #######################################################################################
            examples.append(
                InputExample(
                    "%s-%s" % (set_type, i),
                    text,
                    char_spans["subj"],
                    char_spans["obj"],
                    item["subj_type"],
                    item["obj_type"],
                    item["relation"],
                )
            )

        return examples


def convert_examples_to_features(examples, label_list, tokenizer, max_mention_length, setting='sentence'):
    label_map = {l: i for i, l in enumerate(label_list)}

    def tokenize(text):
        if isinstance(tokenizer, RobertaTokenizer):
            return tokenizer.tokenize(text, add_prefix_space=True)
        else:
            return tokenizer.tokenize(text)

    features = []
    for example in tqdm(examples):
        if setting == 'sentence':
            if example.span_a[1] < example.span_b[1]:
                span_order = ("span_a", "span_b")
            else:
                span_order = ("span_b", "span_a")

            tokens = [tokenizer.cls_token]
            cur = 0
            token_spans = {}
            for span_name in span_order:
                span = getattr(example, span_name)
                tokens += tokenize(example.text[cur : span[0]])
                start = len(tokens)
                tokens.append(HEAD_TOKEN if span_name == "span_a" else TAIL_TOKEN)
                tokens += tokenize(example.text[span[0] : span[1]])
                tokens.append(HEAD_TOKEN if span_name == "span_a" else TAIL_TOKEN)
                token_spans[span_name] = (start, len(tokens))
                cur = span[1]

            tokens += tokenize(example.text[cur:])
            tokens.append(tokenizer.sep_token)

            entity_position_ids = []
            for span_name in ("span_a", "span_b"):
                span = token_spans[span_name]
                position_ids = list(range(span[0], span[1]))[:max_mention_length]
                position_ids += [-1] * (max_mention_length - span[1] + span[0])
                entity_position_ids.append(position_ids)

            entity_segment_ids = [0, 0]
            entity_attention_mask = [1, 1]
        elif setting == 'document':
            text = example.text

            # Insert entity markers into original text. ###########################################
            head_spans = example.span_a
            tail_spans = example.span_b
            all_spans = sorted(head_spans + tail_spans, key=lambda x: x[0])

            intermediate_spans = []
            for idx in range(len(all_spans) - 1):
                span1 = all_spans[idx]
                span2 = all_spans[idx + 1]

                span1_end = span1[1]
                span2_start = span2[0]

                intermediate_span = (span1_end + 1, span2_start)
                intermediate_spans.append(intermediate_span)

            all_spans.extend(intermediate_spans)
            all_spans = sorted(all_spans, key=lambda x: x[0])
            all_spans.append((all_spans[-1][1], len(text)))

            entity_marker_text = []
            for span in all_spans:
                if span in head_spans:
                    text_ = HEAD_TOKEN + ' ' + text[span[0]:span[1]] + ' ' + HEAD_TOKEN
                elif span in tail_spans:
                    text_ = TAIL_TOKEN + ' ' + text[span[0]:span[1]] + ' ' + TAIL_TOKEN
                else:
                    text_ = text[span[0]:span[1]]

                entity_marker_text.append(text_)

            entity_marker_text = ' '.join(entity_marker_text)
            entity_marker_text = ' '.join(entity_marker_text.split()) # Get rid of double-spaces in text.
            #######################################################################################

            # Adjust original spans to incorporate entity marker lengths. #########################
            mention_spans = sorted(head_spans + tail_spans, key=lambda x: x[0])

            adjusted_head_spans = []
            adjusted_tail_spans = []
            adjusted_spans = []

            current_idx = 0
            for span in mention_spans:
                if span in head_spans:
                    head_flag = True
                elif span in tail_spans:
                    head_flag = False

                original_start = span[0]
                original_end = span[1]

                new_start = original_start + (14 * current_idx)
                new_end = original_end + (14 * (current_idx + 1))
                new_span = (new_start, new_end)
                adjusted_spans.append(new_span)

                if head_flag:
                    adjusted_head_spans.append(new_span)
                elif not head_flag:
                    adjusted_tail_spans.append(new_span)

                current_idx += 1
            #######################################################################################

            # Get token spans. ####################################################################
            tokens = [tokenizer.cls_token]

            current_idx = 0
            token_spans = {}
            for span in adjusted_spans:
                if span in adjusted_head_spans:
                    span_name = 'head'
                elif span in adjusted_tail_spans:
                    span_name = 'tail'

                tokens += tokenize(entity_marker_text[current_idx:span[0]])
                start = len(tokens)

                tokens += tokenize(entity_marker_text[span[0]:span[1]])
                end = len(tokens)

                token_span = (start, end)

                try:
                    token_spans[span_name].append(token_span)
                except KeyError:
                    token_spans[span_name] = [token_span]

                current_idx = span[1]

            tokens += tokenize(entity_marker_text[current_idx:])
            tokens.append(tokenizer.sep_token)
            #######################################################################################

            # Get entity positions. We also have to pad the mentions.##############################
            entity_position_ids = []
            for span_name in token_spans:
                spans = token_spans[span_name]
                position_ids = []

                for span in spans:
                    position_id_span = list(range(span[0], span[1]))[:max_mention_length]
                    position_id_span += [-1] * (max_mention_length - span[1] + span[0])
                    position_ids.append(position_id_span)

                entity_position_ids.append(position_ids)

            max_num_mentions = max([len(x) for x in entity_position_ids])
            position_pad = [-1] * max_mention_length
            for idx, position_ids in enumerate(entity_position_ids):
                if len(position_ids) == max_num_mentions:
                    continue

                while len(entity_position_ids[idx]) < max_num_mentions:
                    entity_position_ids[idx].append(position_pad)
            #######################################################################################

        word_ids = tokenizer.convert_tokens_to_ids(tokens)
        word_attention_mask = [1] * len(tokens)
        word_segment_ids = [0] * len(tokens)
        entity_ids = [1, 2]
        entity_segment_ids = [0, 0]
        entity_attention_mask = [1, 1]

        feature = InputFeatures(word_ids=word_ids,
                                word_segment_ids=word_segment_ids,
                                word_attention_mask=word_attention_mask,
                                entity_ids=entity_ids,
                                entity_position_ids=entity_position_ids,
                                entity_segment_ids=entity_segment_ids,
                                entity_attention_mask=entity_attention_mask,
                                label=label_map[example.label])
        features.append(feature)

    return features
