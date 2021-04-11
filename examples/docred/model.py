import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_docred import LukeModelDoc, LukeEntityAwareAttentionModelDoc


class LukeForDocRED(LukeModelDoc):
    def __init__(self, args, num_labels):
        super(LukeForDocRED, self).__init__(args.model_config)

        self.args = args

        self.num_labels = num_labels
        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)
        self.classifier = nn.Linear(args.model_config.hidden_size * 2, num_labels, False)

        self.apply(self.init_weights)

    def get_head_tail_representations(self, sequence_output, head_tail_idxs, entity_position_ids):
        """
        Get representations for (head, tail) pairs. You should end up with (batch_size * total_num_head_tail_pairs) samples.
        """
        all_head_representations = []
        all_tail_representations = []
        all_relation_representations = []

        for batch_idx, _ in enumerate(head_tail_idxs):
            head_representations = []
            tail_representations = []
            relation_representations = []

            encoded_text = sequence_output[batch_idx]
            head_tail_pairs = head_tail_idxs[batch_idx]
            entities = entity_position_ids[batch_idx]

            for pair in head_tail_pairs:
                head_embeddings = []
                tail_embeddings = []

                head_idx = pair[0]
                tail_idx = pair[1]

                head_entity_positions = entities[head_idx]
                tail_entity_positions = entities[tail_idx]

                for head_entity_position in head_entity_positions:
                    valid_position = [idx for idx in head_entity_position if (idx != -1) and (idx < 512)]
                    head_embeddings_ = encoded_text[valid_position]
                    head_embeddings.append(torch.sum(head_embeddings_, dim=0, keepdim=True))

                for tail_entity_position in tail_entity_positions:
                    valid_position = [idx for idx in tail_entity_position if (idx != -1) and (idx < 512)]
                    tail_embeddings_ = encoded_text[valid_position]
                    tail_embeddings.append(torch.sum(tail_embeddings_, dim=0, keepdim=True))

                head_embeddings = torch.cat(head_embeddings, dim=0)
                tail_embeddings = torch.cat(tail_embeddings, dim=0)

                head_entity_embedding = torch.sum(head_embeddings, dim=0, keepdim=True)
                tail_entity_embedding = torch.sum(tail_embeddings, dim=0, keepdim=True)

                relation_representation = torch.cat([head_entity_embedding, tail_entity_embedding], dim=1)

                head_representations.append(head_entity_embedding)
                tail_representations.append(tail_entity_embedding)
                relation_representations.append(relation_representation)

            all_head_representations.append(head_representations)
            all_tail_representations.append(tail_representations)
            all_relation_representations.append(relation_representations)

        return all_head_representations, all_tail_representations, all_relation_representations


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

        heads, tails, relations = self.get_head_tail_representations(sequence_output, head_tail_idxs, entity_position_ids)

        feature_vector = torch.cat(sum(relations, []), dim=0)
        feature_vector = self.dropout(feature_vector)

        logits = self.classifier(feature_vector)

        if label is None:
            return logits

        labels = torch.tensor(sum(label, [])).to(self.args.device)

        return (F.cross_entropy(logits, labels),)

        # # encoder_outputs[0] -> context representations
        # # encoder_outputs[1] -> entity representations

        # # Concatenate head and tail entity hidden states.
        # # feature_vector.shape = (batch_size, hidden_dim * 2)
        # feature_vector = torch.cat([encoder_outputs[1][:, 0, :], encoder_outputs[1][:, 1, :]], dim=1)
        # feature_vector = self.dropout(feature_vector)

        # logits = self.classifier(feature_vector)

        # if label is None:
        #     return logits

        # return (F.cross_entropy(logits, label),)
