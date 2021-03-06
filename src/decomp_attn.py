from typing import Dict, Optional, List, Any

import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("decomp_attn")
class DecomposableAttentionModel(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 attend_feedforward: FeedForward,
                 similarity_function: SimilarityFunction,
                 compare_feedforward: FeedForward,
                 aggregate_feedforward: FeedForward,
                 use_sentiment: bool,
                 use_tfidf: bool,
                 headline_encoder: Optional[Seq2SeqEncoder] = None,
                 body_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DecomposableAttentionModel, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward
        self._headline_encoder = headline_encoder
        self._body_encoder = body_encoder or headline_encoder

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        self.use_sentiment = use_sentiment
        self.use_tfidf = use_tfidf

        self._accuracy = CategoricalAccuracy()

        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                headline: Dict[str, torch.LongTensor],
                body: Dict[str, torch.LongTensor],
                headline_sentiment=None,
                body_sentiment=None,
                tfidf=None,
                stance: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        embedded_headline = self._text_field_embedder(headline)
        embedded_body = self._text_field_embedder(body)
        headline_mask = get_text_field_mask(headline).float()
        body_mask = get_text_field_mask(body).float()

        if self._headline_encoder:
            embedded_headline = self._headline_encoder(embedded_headline, headline_mask)
        if self._body_encoder:
            embedded_body = self._body_encoder(embedded_body, body_mask)

        projected_headline = self._attend_feedforward(embedded_headline)
        projected_body = self._attend_feedforward(embedded_body)
        # Shape: (batch_size, headline_length, body_length)
        similarity_matrix = self._matrix_attention(projected_headline, projected_body)

        # Shape: (batch_size, headline_length, body_length)
        p2h_attention = masked_softmax(similarity_matrix, body_mask)
        # Shape: (batch_size, headline_length, embedding_dim)
        attended_body = weighted_sum(embedded_body, p2h_attention)

        # Shape: (batch_size, body_length, headline_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), headline_mask)
        # Shape: (batch_size, body_length, embedding_dim)
        attended_headline = weighted_sum(embedded_headline, h2p_attention)

        headline_compare_input = torch.cat([embedded_headline, attended_body], dim=-1)
        body_compare_input = torch.cat([embedded_body, attended_headline], dim=-1)

        compared_headline = self._compare_feedforward(headline_compare_input)
        compared_headline = compared_headline * headline_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_headline = compared_headline.sum(dim=1)

        compared_body = self._compare_feedforward(body_compare_input)
        compared_body = compared_body * body_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_body = compared_body.sum(dim=1)

        aggregate_input = torch.cat([compared_headline, compared_body], dim=-1)
        if self.use_sentiment:
            aggregate_input = torch.cat([aggregate_input, headline_sentiment, body_sentiment], dim=-1)
        if self.use_tfidf:
            aggregate_input = torch.cat([aggregate_input, tfidf], dim=-1)
            

        label_logits = self._aggregate_feedforward(aggregate_input)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits,
                       "label_probs": label_probs,
                       "h2p_attention": h2p_attention,
                       "p2h_attention": p2h_attention}

        if stance is not None:
            loss = self._loss(label_logits, stance.long().view(-1))
            self._accuracy(label_logits, stance)
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["headline_tokens"] = [x["headline_tokens"] for x in metadata]
            output_dict["body_tokens"] = [x["body_tokens"] for x in metadata]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }
