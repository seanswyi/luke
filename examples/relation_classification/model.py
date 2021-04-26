import torch
import torch.nn as nn
import torch.nn.functional as F

from .at_loss import ATLoss
from luke.model import LukeEntityAwareAttentionModel


class LukeForRelationClassification(LukeEntityAwareAttentionModel):
    def __init__(self, args, num_labels):
        super(LukeForRelationClassification, self).__init__(args.model_config)

        self.args = args
        self.block_size = 64

        self.num_labels = num_labels
        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)

        if args.classifier == 'linear':
            self.classifier = nn.Linear(args.model_config.hidden_size * 2, num_labels, False)
        elif args.classifier == 'bilinear':
            self.classifier = nn.Linear(in_features=(args.model_config.hidden_size * self.block_size), out_features=num_labels)

        self.head_extractor = nn.Linear(in_features=args.model_config.hidden_size, out_features=args.model_config.hidden_size)
        self.tail_extractor = nn.Linear(in_features=args.model_config.hidden_size, out_features=args.model_config.hidden_size)

        if args.atloss:
            self.loss_function = ATLoss()
        elif not args.atloss:
            self.loss_function = nn.CrossEntropyLoss()

        self.apply(self.init_weights)

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
    ):
        encoder_outputs = super(LukeForRelationClassification, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        # Concatenate head and tail entity hidden states.
        # feature_vector.shape = (batch_size, hidden_dim * 2)

        if self.args.classifier == 'linear':
            feature_vector = torch.cat([encoder_outputs[1][:, 0, :], encoder_outputs[1][:, 1, :]], dim=1)
            feature_vector = self.dropout(feature_vector)
        elif self.args.classifier == 'bilinear':
            z_s = torch.tanh(self.head_extractor(encoder_outputs[1][:, 0, :]))
            z_o = torch.tanh(self.tail_extractor(encoder_outputs[1][:, 1, :]))

            b1 = z_s.view(-1, self.args.model_config.hidden_size // self.block_size, self.block_size)
            b2 = z_o.view(-1, self.args.model_config.hidden_size // self.block_size, self.block_size)
            bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.args.model_config.hidden_size * self.block_size)

            feature_vector = self.dropout(bl)

        logits = self.classifier(feature_vector)

        if label is None:
            return logits

        if self.args.atloss:
            label = torch.zeros(len(label), self.num_labels).scatter_(1, label.unsqueeze(1), 1.0)

        return (self.loss_function(logits, label.to(self.args.device)),)
