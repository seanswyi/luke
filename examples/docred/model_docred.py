import logging
import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertIntermediate,
    BertLayerNorm,
    BertOutput,
    BertPooler,
    BertSelfOutput,
)
from transformers.modeling_roberta import RobertaEmbeddings

from .process_long_seq import process_long_input


logger = logging.getLogger(__name__)


def pad_lists(max_num, list_of_lists):
    pad_template = [-1] * len(list_of_lists[0][0])

    for idx, l in enumerate(list_of_lists):
        needed_padding = max_num - len(l)
        list_of_lists[idx].extend([pad_template] * needed_padding)

    return list_of_lists


class LukeConfig(BertConfig):
    def __init__(
        self, vocab_size: int, entity_vocab_size: int, bert_model_name: str, entity_emb_size: int = None, **kwargs
    ):
        super(LukeConfig, self).__init__(vocab_size, **kwargs)

        self.entity_vocab_size = entity_vocab_size
        self.bert_model_name = bert_model_name
        self.output_attentions = True

        if entity_emb_size is None:
            self.entity_emb_size = self.hidden_size
        else:
            self.entity_emb_size = entity_emb_size


class EntityEmbeddings(nn.Module):
    def __init__(self, config: LukeConfig):
        super(EntityEmbeddings, self).__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, entity_ids, position_ids, token_type_ids, head_tail_idxs):
        all_embeddings = []

        expanded_entity_ids = []
        for head_tail_idx in head_tail_idxs:
            entity_ids_ = torch.stack(len(head_tail_idx) * [entity_ids[0]]).to('cuda')
            expanded_entity_ids.append(entity_ids_)

        expanded_token_type_ids = []
        for head_tail_idx in head_tail_idxs:
            token_type_ids_ = torch.stack(len(head_tail_idx) * [token_type_ids[0]]).to('cuda')
            expanded_token_type_ids.append(token_type_ids_)

        for batch_idx, _ in enumerate(entity_ids):
            batch_entity_ids = expanded_entity_ids[batch_idx]
            batch_token_type_ids = expanded_token_type_ids[batch_idx]
            batch_position_ids = position_ids[batch_idx]
            batch_head_tail_idxs = head_tail_idxs[batch_idx]

            batch_entity_embeddings = self.entity_embeddings(batch_entity_ids)
            if self.config.entity_emb_size != self.config.hidden_size:
                batch_entity_embeddings = self.entity_embedding_dense(batch_entity_embeddings)

            token_type_id_embeddings = self.token_type_embeddings(batch_token_type_ids)

            max_num = len(max(batch_position_ids, key=len))
            batch_position_ids = pad_lists(max_num=max_num, list_of_lists=batch_position_ids)

            position_embedding_mask = []
            batch_position_embeddings = []
            for mentions in batch_position_ids:
                position_embeddings = self.position_embeddings(torch.tensor(mentions).clamp(min=0).to('cuda'))
                batch_position_embeddings.append(position_embeddings)

                mention_mask = []
                for mention in mentions:
                    mask = [1 if x != -1 else 0 for x in mention]
                    mention_mask.append(mask)

                position_embedding_mask.append(mention_mask)

            batch_position_embeddings = torch.stack(batch_position_embeddings, dim=0)
            position_embedding_mask = torch.FloatTensor(position_embedding_mask).unsqueeze(-1).to('cuda')
            batch_position_embeddings = batch_position_embeddings * position_embedding_mask

            batch_position_embeddings = batch_position_embeddings.sum(dim=2)
            normalization_factor = position_embedding_mask.sum(dim=2).clamp(min=1e-7)
            batch_position_embeddings = batch_position_embeddings / normalization_factor

            batch_head_tail_idxs = torch.tensor(batch_head_tail_idxs).to('cuda')
            selected_position_embeddings = batch_position_embeddings[batch_head_tail_idxs]

            head_embeddings = batch_entity_embeddings[:, 0] # [num_head_tail_pairs, 1024]
            tail_embeddings = batch_entity_embeddings[:, 1]

            head_position_embeddings = selected_position_embeddings[:, 0] # [num_head_tail_pairs, num_mentions, 30, 1024]
            tail_position_embeddings = selected_position_embeddings[:, 1]

            head_position_embeddings = head_position_embeddings.sum(dim=1)
            tail_position_embeddings = tail_position_embeddings.sum(dim=1)
            try:
                head_entity_embeddings = head_embeddings + head_position_embeddings + token_type_id_embeddings[:, 0]
                tail_entity_embeddings = tail_embeddings + tail_position_embeddings + token_type_id_embeddings[:, 1]
            except:
                import pdb; pdb.set_trace()

            entity_embeddings = torch.stack([head_entity_embeddings, tail_entity_embeddings], dim=1)
            entity_embeddings = self.LayerNorm(entity_embeddings)
            entity_embeddings = self.dropout(entity_embeddings)

            all_embeddings.append(entity_embeddings)

        return all_embeddings


class LukeModelDoc(nn.Module):
    def __init__(self, config: LukeConfig):
        super(LukeModelDoc, self).__init__()
        self.config = config

        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
            self.embeddings = RobertaEmbeddings(config)
            self.embeddings.token_type_embeddings.requires_grad = False
        else:
            self.embeddings = BertEmbeddings(config)
        self.entity_embeddings = EntityEmbeddings(config)

    def forward(
        self,
        word_ids: torch.LongTensor,
        word_segment_ids: torch.LongTensor,
        word_attention_mask: torch.LongTensor,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        entity_segment_ids: torch.LongTensor = None,
        entity_attention_mask: torch.LongTensor = None,
        head_tail_idxs: torch.LongTensor = None
    ):
        word_seq_size = word_ids.size(1)

        start_tokens = torch.tensor([self.args.tokenizer.cls_token_id]).to(word_ids)
        end_tokens = torch.tensor([self.args.tokenizer.sep_token_id]).to(word_ids)
        start_length = start_tokens.shape[0]
        end_length = end_tokens.shape[0]

        if word_seq_size <= 512:
            embedding_output = self.embeddings(word_ids)
            attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask) # [batch_size, 1, 1, word_size]

            if entity_ids is not None:
                entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids, head_tail_idxs)
                import pdb; pdb.set_trace()
                embedding_output = torch.cat([embedding_output, entity_embedding_output], dim=1)

            encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask, head_mask=([None] * self.config.num_hidden_layers))

            sequence_output = encoder_outputs[0]
            attention_output = encoder_outputs[-1][-1]

            pooled_output = self.pooler(sequence_output)

            word_sequence_output = sequence_output[:, :word_seq_size, :]

            if entity_ids:
                entity_sequence_output = sequence_output[:, word_seq_size:, :]
                output = (word_sequence_output, entity_sequence_output, pooled_output, attention_output)
            else:
                output = (word_sequence_output, pooled_output, attention_output)
        elif word_seq_size > 512:
            new_word_ids = []
            new_attention_mask = []
            # new_word_segment_ids = []
            num_segments = []

            sequence_lengths = word_attention_mask.sum(1).detach().cpu().numpy().tolist()
            for idx, sequence_length in enumerate(sequence_lengths):
                if sequence_length <= 512:
                    new_word_ids.append(word_ids[idx, :512])
                    new_attention_mask.append(word_attention_mask[idx, :512])
                    num_segments.append(1)
                elif sequence_length > 512:
                    word_ids1 = torch.cat([word_ids[idx, :512 - end_length], end_tokens], dim=-1)
                    word_ids2 = torch.cat([start_tokens, word_ids[idx, (sequence_length - 512 + start_length):sequence_length]], dim=-1)

                    word_attention_mask1 = word_attention_mask[idx, :512]
                    word_attention_mask2 = word_attention_mask[idx, (sequence_length - 512):sequence_length]

                    # word_segment_ids1 = word_segment_ids[idx, :512]
                    # word_segment_ids2 = word_segment_ids[idx, (sequence_length - 512):sequence_length]

                    new_word_ids.extend([word_ids1, word_ids2])
                    new_attention_mask.extend([word_attention_mask1, word_attention_mask2])
                    # new_word_segment_ids.extend([word_segment_ids1, word_segment_ids2])
                    num_segments.append(2)

            word_ids = torch.stack(new_word_ids, dim=0)
            word_attention_mask = torch.stack(new_attention_mask, dim=0)
            # word_segment_ids = torch.stack(new_word_segment_ids, dim=0)

            embedding_output = self.embeddings(word_ids)
            attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask) # [batch_size, 1, 1, word_size]

            if entity_ids is not None:
                entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
                embedding_output = torch.cat([embedding_output, entity_embedding_output], dim=1)

            encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask, head_mask=([None] * self.config.num_hidden_layers))

            sequence_output = encoder_outputs[0]
            attention_output = encoder_outputs[-1][-1]

            current_idx = 0
            new_sequence_output = []
            new_attention_output = []

            for num_segment, sequence_length in zip(num_segments, sequence_lengths):
                if num_segment == 1:
                    output = F.pad(sequence_output[current_idx], (0, 0, 0, word_seq_size - 512))
                    attention = F.pad(attention_output[current_idx], (0, word_seq_size - 512, 0, word_seq_size - 512))

                    new_sequence_output.append(output)
                    new_attention_output.append(attention)
                elif num_segment == 2:
                    output1 = sequence_output[current_idx][:512 - end_length]
                    mask1 = word_attention_mask[current_idx][:512 - end_length]
                    attention1 = attention_output[current_idx][:, :512 - end_length, :512 - end_length]
                    output1 = F.pad(output1, (0, 0, 0, word_seq_size - 512 + end_length))
                    mask1 = F.pad(mask1, (0, word_seq_size - 512 + end_length))
                    attention1 = F.pad(attention1, (0, word_seq_size - 512 + end_length, 0, word_seq_size - 512 + end_length))

                    output2 = sequence_output[current_idx + 1][start_length:]
                    mask2 = word_attention_mask[current_idx + 1][start_length:]
                    attention2 = attention_output[current_idx + 1][:, start_length:, start_length:]
                    output2 = F.pad(output2, (0, 0, sequence_length - 512 + start_length, word_seq_size - sequence_length))
                    mask2 = F.pad(mask2, (sequence_length - 512 + start_length, word_seq_size - sequence_length))
                    attention2 = F.pad(attention2, [sequence_length - 512 + start_length, word_seq_size - sequence_length, sequence_length - 512 + start_length, word_seq_size - sequence_length])

                    mask = mask1 + mask2 + 1e-10
                    output = (output1 + output2) / mask.unsqueeze(-1).float()
                    attention = attention1 + attention2
                    attention = attention / (attention.sum(-1, keepdim=True) + 1e-10)

                    new_sequence_output.append(output)
                    new_attention_output.append(attention)

                current_idx += num_segment

            sequence_output = torch.stack(new_sequence_output, dim=0)
            attention_output = torch.stack(new_attention_output, dim=0)

            assert sequence_output.shape[1] == attention_output.shape[-1] == word_seq_size

            pooled_output = self.pooler(sequence_output)

            word_sequence_output = sequence_output[:, :word_seq_size, :]

            if entity_ids:
                entity_sequence_output = sequence_output[:, word_seq_size:, :]
                output = (word_sequence_output, entity_sequence_output, pooled_output, attention_output)
            else:
                output = (word_sequence_output, pooled_output, attention_output)

        return output

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            if module.embedding_dim == 1:  # embedding for bias parameters
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_bert_weights(self, state_dict: Dict[str, torch.Tensor]):
        state_dict = state_dict.copy()
        for key in list(state_dict.keys()):
            new_key = key.replace("gamma", "weight").replace("beta", "bias")
            if new_key.startswith("roberta."):
                new_key = new_key[8:]
            elif new_key.startswith("bert."):
                new_key = new_key[5:]

            if key != new_key:
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self, prefix="")
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    self.__class__.__name__, sorted(unexpected_keys)
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(self.__class__.__name__, "\n\t".join(error_msgs))
            )

    def _compute_extended_attention_mask(
        self, word_attention_mask: torch.LongTensor, entity_attention_mask: torch.LongTensor
    ):
        attention_mask = word_attention_mask

        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=1)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


class LukeEntityAwareAttentionModelDoc(LukeModelDoc):
    def __init__(self, config):
        super(LukeEntityAwareAttentionModelDoc, self).__init__(config)
        self.config = config
        self.encoder = EntityAwareEncoder(config)

    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        token_type_ids,
        head_tail_idxs
    ):
        word_ids = word_ids.to(self.args.device)
        word_segment_ids = word_segment_ids.to(self.args.device)
        word_attention_mask = word_attention_mask.to(self.args.device)
        entity_ids = entity_ids.to(self.args.device)
        entity_segment_ids = entity_segment_ids.to(self.args.device)
        entity_attention_mask = entity_attention_mask.to(self.args.device)

        word_embeddings = self.embeddings(word_ids, word_segment_ids) # [batch_size, seq_len, emb_dim]
        entity_embeddings = self.entity_embeddings(entity_ids, entity_position_ids, token_type_ids, head_tail_idxs)

        # entity_embeddings = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids) # [batch_size, entity_size, emb_dim]
        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

        encoded_output = self.encoder(word_embeddings, entity_embeddings, attention_mask)

        return encoded_output

    def load_state_dict(self, state_dict, *args, **kwargs):
        new_state_dict = state_dict.copy()

        for num in range(self.config.num_hidden_layers):
            for attr_name in ("weight", "bias"):
                if f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}" not in state_dict:
                    new_state_dict[f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}"] = state_dict[
                        f"encoder.layer.{num}.attention.self.query.{attr_name}"
                    ]
                if f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}" not in state_dict:
                    new_state_dict[f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}"] = state_dict[
                        f"encoder.layer.{num}.attention.self.query.{attr_name}"
                    ]
                if f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}" not in state_dict:
                    new_state_dict[f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}"] = state_dict[
                        f"encoder.layer.{num}.attention.self.query.{attr_name}"
                    ]

        kwargs["strict"] = False
        super(LukeEntityAwareAttentionModelDoc, self).load_state_dict(new_state_dict, *args, **kwargs)


class EntityAwareSelfAttention(nn.Module):
    def __init__(self, config):
        super(EntityAwareSelfAttention, self).__init__()
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        new_x = x.view(*new_x_shape).permute(0, 2, 1, 3)

        return new_x

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        """
        word_hidden_states.shape = [batch_size, num_words, hidden_dim]
        entity_hidden_states.shape = [batch_size, num_entities, hidden_dim]
        """
        word_size = word_hidden_states.size(1)

        all_word_representations = []
        all_entity_representations = []

        for batch_idx, _ in enumerate(word_hidden_states):
            word_embeddings = word_hidden_states[batch_idx]
            entity_embeddings = entity_hidden_states[batch_idx]
            batch_attention_mask = attention_mask[batch_idx]

            num_head_tail_pairs = entity_embeddings.shape[0]
            expanded_word_embeddings = word_embeddings.expand(num_head_tail_pairs, word_size, self.config.hidden_size)

            w2w_query_layer = self.transpose_for_scores(self.query(word_embeddings.unsqueeze(0)))
            w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_embeddings.unsqueeze(0)))
            e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_embeddings))
            e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_embeddings))

            key_layer = self.transpose_for_scores(self.key(torch.cat([expanded_word_embeddings, entity_embeddings], dim=1)))

            w2w_key_layer = key_layer[:, :, :word_size, :]
            e2w_key_layer = key_layer[:, :, :word_size, :]
            w2e_key_layer = key_layer[:, :, word_size:, :]
            e2e_key_layer = key_layer[:, :, word_size:, :]

            w2w_attention_scores = torch.matmul(w2w_query_layer, w2w_key_layer.transpose(-1, -2))
            w2e_attention_scores = torch.matmul(w2e_query_layer, w2e_key_layer.transpose(-1, -2))
            e2w_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2))
            e2e_attention_scores = torch.matmul(e2e_query_layer, e2e_key_layer.transpose(-1, -2))

            del w2w_query_layer
            del w2e_query_layer
            del e2w_query_layer
            del e2e_query_layer
            del w2w_key_layer
            del w2e_key_layer
            del e2w_key_layer
            del e2e_key_layer
            del key_layer
            torch.cuda.empty_cache()

            word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=-1)

            del w2w_attention_scores
            del w2e_attention_scores
            torch.cuda.empty_cache()

            entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=-1)

            del e2w_attention_scores
            del e2e_attention_scores
            torch.cuda.empty_cache()

            attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2)

            del word_attention_scores
            del entity_attention_scores
            torch.cuda.empty_cache()

            attention_scores = (attention_scores / math.sqrt(self.attention_head_size)) + batch_attention_mask

            del batch_attention_mask
            torch.cuda.empty_cache()

            attention_probs = F.softmax(attention_scores, dim=-1)

            del attention_scores
            torch.cuda.empty_cache()

            value_layer = self.transpose_for_scores(self.value(torch.cat([expanded_word_embeddings, entity_embeddings], dim=1)))

            del expanded_word_embeddings
            del entity_embeddings
            torch.cuda.empty_cache()

            context_representation = torch.matmul(attention_probs, value_layer).permute(0, 2, 1, 3).contiguous()

            del attention_probs
            del value_layer
            torch.cuda.empty_cache()

            new_context_representation_shape = context_representation.shape[:-2] + (self.all_head_size,)
            context_representation = context_representation.view(*new_context_representation_shape)

            word_representations = context_representation[:, :word_size, :]
            entity_representations = context_representation[:, word_size:, :]

            all_word_representations.append(word_representations)
            all_entity_representations.append(entity_representations)


        import pdb; pdb.set_trace()

                # attention_probs = F.softmax(attention_scores, dim=-1)
                # attention_probs = self.dropout(attention_probs)

                # value_layer = self.transpose_for_scores(self.value(torch.cat([encoded_text, pair], dim=0)))

                # context_layer = torch.matmul(attention_probs, value_layer)
                # context_layer = context_layer.permute(1, 0, 2).contiguous()
                # new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
                # context_layer = context_layer.view(*new_context_layer_shape)

                # word_representations = context_layer[:word_size, :]
                # entity_representations = context_layer[word_size:, :]

                # all_word_representations.append(word_representations)
                # all_entity_representations.append(entity_representations)

        #         import pdb; pdb.set_trace()

        # w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states)) # [batch_size, num_attention_heads, word_size, head_size]
        # w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states)) # [batch_size, num_attention_heads, word_size, head_size]
        # e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states)) # [batch_size, num_attention_heads, entity_size, head_size]
        # e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states)) # [batch_size, num_attention_heads, entity_size, head_size]

        # key_layer = self.transpose_for_scores(self.key(torch.cat([word_hidden_states, entity_hidden_states], dim=1)))

        # w2w_key_layer = key_layer[:, :, :word_size, :]
        # e2w_key_layer = key_layer[:, :, :word_size, :]
        # w2e_key_layer = key_layer[:, :, word_size:, :]
        # e2e_key_layer = key_layer[:, :, word_size:, :]

        # w2w_attention_scores = torch.matmul(w2w_query_layer, w2w_key_layer.transpose(-1, -2)) # [batch_size, num_attention_heads, word_size, word_size]
        # w2e_attention_scores = torch.matmul(w2e_query_layer, w2e_key_layer.transpose(-1, -2)) # [batch_size, num_attention_heads, word_size, entity_size]
        # e2w_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2)) # [batch_size, num_attention_heads, entity_size, word_size]
        # e2e_attention_scores = torch.matmul(e2e_query_layer, e2e_key_layer.transpose(-1, -2)) # [batch_size, num_attention_heads, entity_size, entity_size]

        # word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=3) # [batch_size, num_attention_heads, word_size, word_size + entity_size]
        # entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=3) # [batch_size, num_attention_heads, entity_size, word_size + entity_size]
        # attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2) # [batch_size, num_attention_heads, word_size + entity_size, word_size + enitity_size]

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores = attention_scores + attention_mask

        # attention_probs = F.softmax(attention_scores, dim=-1)
        # attention_probs = self.dropout(attention_probs)

        # value_layer = self.transpose_for_scores(
        #     self.value(torch.cat([word_hidden_states, entity_hidden_states], dim=1))
        # )
        # context_layer = torch.matmul(attention_probs, value_layer) # [batch_size, num_attention_heads, word_size + entity_size, attention_size]

        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)

        return all_word_representations, all_entity_representations


class EntityAwareAttention(nn.Module):
    def __init__(self, config):
        super(EntityAwareAttention, self).__init__()
        self.self = EntityAwareSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_self_output, entity_self_output = self.self(word_hidden_states, entity_hidden_states, attention_mask)
        import pdb; pdb.set_trace()
        hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1) # [batch_size, word_size + entity_size, hidden_dim]
        self_output = torch.cat([word_self_output, entity_self_output], dim=1)

        output = self.output(self_output, hidden_states)

        return output[:, : word_hidden_states.size(1), :], output[:, word_hidden_states.size(1) :, :]


class EntityAwareLayer(nn.Module):
    def __init__(self, config):
        super(EntityAwareLayer, self).__init__()

        self.attention = EntityAwareAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_attention_output, entity_attention_output = self.attention(
            word_hidden_states, entity_hidden_states, attention_mask
        )
        import pdb; pdb.set_trace()
        attention_output = torch.cat([word_attention_output, entity_attention_output], dim=1)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output[:, : word_hidden_states.size(1), :], layer_output[:, word_hidden_states.size(1) :, :]


class EntityAwareEncoder(nn.Module):
    def __init__(self, config):
        super(EntityAwareEncoder, self).__init__()
        self.layer = nn.ModuleList([EntityAwareLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        for layer_module in self.layer:
            word_hidden_states, entity_hidden_states = layer_module(
                word_hidden_states, entity_hidden_states, attention_mask
            )
        return word_hidden_states, entity_hidden_states
