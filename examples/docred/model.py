import torch
import torch.nn as nn
import torch.nn.functional as F

from .at_loss import ATLoss
from .model_docred import LukeModelDoc, LukeEntityAwareAttentionModelDoc
from .process_long_seq import process_long_input


class LukeForDocRED(LukeModelDoc):
    def __init__(self, args, num_labels):
        super(LukeForDocRED, self).__init__(args.model_config)

        self.args = args
        self.block_size = 64

        self.num_labels = num_labels
        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)

        if self.args.classifier == 'linear':
            self.classifier = nn.Linear(args.model_config.hidden_size * 2, num_labels, False)
        elif self.args.classifier == 'bilinear':
            self.classifier = nn.Linear(args.model_config.hidden_size * self.block_size, num_labels)

        if self.args.lop:
            self.head_extractor = nn.Linear(args.model_config.hidden_size * 2, args.model_config.hidden_size)
            self.tail_extractor = nn.Linear(args.model_config.hidden_size * 2, args.model_config.hidden_size)
        elif not self.args.lop:
            self.head_extractor = nn.Linear(args.model_config.hidden_size, args.model_config.hidden_size)
            self.tail_extractor = nn.Linear(args.model_config.hidden_size, args.model_config.hidden_size)

        self.apply(self.init_weights)

        self.at_loss = ATLoss()

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_head_tail_representations(self, sequence_output, attention, head_tail_idxs, entity_position_ids):
        """
        Get representations for (head, tail) pairs. You should end up with (batch_size * total_num_head_tail_pairs) samples.
        """
        all_head_representations = []
        all_tail_representations = []
        all_local_attentions = []

        for batch_idx, _ in enumerate(head_tail_idxs):
            head_representations = []
            tail_representations = []
            local_attentions = []

            encoded_text = sequence_output[batch_idx]
            attention_output = attention[batch_idx]
            head_tail_pairs = head_tail_idxs[batch_idx]
            entities = entity_position_ids[batch_idx]

            for pair in head_tail_pairs:
                head_embeddings = []
                tail_embeddings = []
                head_attentions = []
                tail_attentions = []

                head_idx = pair[0]
                tail_idx = pair[1]

                head_entity_positions = entities[head_idx]
                tail_entity_positions = entities[tail_idx]

                for head_entity_position in head_entity_positions:
                    valid_position = [idx for idx in head_entity_position if idx != -1]
                    head_embeddings_ = encoded_text[valid_position]
                    try:
                        head_attentions_ = attention_output[:, valid_position]
                    except RuntimeError:
                        import pdb; pdb.set_trace()
                    head_embeddings.append(torch.sum(head_embeddings_, dim=0, keepdim=True))
                    head_attentions.append(head_attentions_)

                for tail_entity_position in tail_entity_positions:
                    valid_position = [idx for idx in tail_entity_position if idx != -1]
                    tail_embeddings_ = encoded_text[valid_position]
                    tail_attentions_ = attention_output[:, valid_position]
                    tail_embeddings.append(torch.sum(tail_embeddings_, dim=0, keepdim=True))
                    tail_attentions.append(tail_attentions_)

                head_embeddings = torch.cat(head_embeddings, dim=0)
                tail_embeddings = torch.cat(tail_embeddings, dim=0)

                head_entity_embedding = torch.sum(head_embeddings, dim=0, keepdim=True)
                tail_entity_embedding = torch.sum(tail_embeddings, dim=0, keepdim=True)

                head_attentions = torch.cat(head_attentions, dim=1).mean(1)
                tail_attentions = torch.cat(tail_attentions, dim=1).mean(1)
                head_tail_attentions = (head_attentions * tail_attentions).mean(0, keepdim=True)
                head_tail_attentions = head_tail_attentions / (head_tail_attentions.sum(1, keepdim=True) + 1e-10)
                local_attention = torch.matmul(head_tail_attentions, encoded_text)

                head_representations.append(head_entity_embedding)
                tail_representations.append(tail_entity_embedding)
                local_attentions.append(local_attention)

            all_head_representations.append(head_representations)
            all_tail_representations.append(tail_representations)
            all_local_attentions.append(local_attentions)

        return all_head_representations, all_tail_representations, all_local_attentions

    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        label=None,
        head_tail_idxs=None
    ):
        word_ids = word_ids.to(self.args.device)
        word_segment_ids = word_segment_ids.to(self.args.device)
        word_attention_mask = word_attention_mask.to(self.args.device)

        encoder_outputs = super(LukeForDocRED, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            None,
            None,
            None,
            None,
            None
        )

        sequence_output = encoder_outputs[0]
        attention_output = encoder_outputs[-1]

        heads, tails, attentions = self.get_head_tail_representations(sequence_output, attention_output, head_tail_idxs, entity_position_ids)

        all_heads = torch.cat(sum(heads, []), dim=0) # [batch_size * num_head_tails, hidden_dim]
        all_tails = torch.cat(sum(tails, []), dim=0) # [batch_size * num_head_tails, hidden_dim]
        all_attentions = torch.cat(sum(attentions, []), dim=0) # [batch_size * num_head_tails, hidden_dim]

        if self.args.classifier == 'linear':
            feature_vector = self.dropout(torch.cat([all_heads, all_tails], dim=1))
        elif self.args.classifier == 'bilinear':
            if self.args.lop:
                all_heads = torch.cat([all_heads, all_attentions], dim=1)
                all_tails = torch.cat([all_tails, all_attentions], dim=1)

            z_s = torch.tanh(self.head_extractor(all_heads)) # [batch_size * num_head_tails, hidden_dim]
            z_o = torch.tanh(self.tail_extractor(all_tails)) # [batch_size * num_head_tails, hidden_dim]

            b1 = z_s.view(-1, self.args.model_config.hidden_size // self.block_size, self.block_size) # [batch_size * num_head_tails, hidden_dim / block_size, block_size]
            b2 = z_o.view(-1, self.args.model_config.hidden_size // self.block_size, self.block_size) # [batch_size * num_head_tails, hidden_dim / block_size, block_size]
            bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.args.model_config.hidden_size * self.block_size) # [batch_size * num_head_tails, hidden_dim * block_size]

            feature_vector = bl

        logits = self.classifier(feature_vector)

        if label is None:
            return logits

        labels = torch.tensor(sum(label, [])).to(self.args.device)

        if self.args.at_loss:
            one_hot_labels = torch.zeros(size=(labels.shape[0], self.num_labels)).to(self.args.device)

            for idx, label in enumerate(labels):
                label_value = label.cpu().item()
                one_hot_labels[idx][label_value] = 1

            loss = self.at_loss(logits, one_hot_labels)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss,)
