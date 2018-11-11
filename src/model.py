from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("baseline_model")
class BasicSequenceModel(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 headline_encoder: Seq2VecEncoder,
                 body_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BasicSequenceModel, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size('labels')
        self.headline_encoder = headline_encoder
        self.body_encoder = body_encoder
        self.classifier_feedforward = classifier_feedforward

        self.metrics = {
                "accuracy": CategoricalAccuracy(),
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self, headline, body, stance=None):
        embedded_headline = self.text_field_embedder(headline)
        headline_mask = util.get_text_field_mask(headline)
        encoded_headline = self.headline_encoder(embedded_headline, headline_mask)

        embedded_body = self.text_field_embedder(body)
        body_mask = util.get_text_field_mask(body)
        encoded_body = self.body_encoder(embedded_body, body_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_headline, encoded_body], dim=-1))
        output_dict = {'logits': logits}
        if stance is not None:
            loss = self.loss(logits, stance)
            for metric in self.metrics.values():
                metric(logits, stance)
            output_dict["loss"] = loss

        return output_dict

    def decode(self, output_dict):
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        stance = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['stance'] = stance
        return output_dict

    def get_metrics(self, reset):
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
