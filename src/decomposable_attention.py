from typing import Dict, Optional, List, Any

import torch
import numpy

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("decomp_attention")
class DecomposableAttention(Model):
    """
    This ``Model`` implements the Decomposable Attention model described in `"A Decomposable
    Attention Model for Natural Language Inference"
    <https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_
    by Parikh et al., 2016, with some optional enhancements before the decomposable attention
    actually happens.  Parikh's original model allowed for computing an "intra-sentence" attention
    before doing the decomposable entailment step.  We generalize this to any
    :class:`Seq2SeqEncoder` that can be applied to the headline and/or the body before
    computing entailment.

    The basic outline of this model is to get an embedded representation of each word in the
    headline and body, align words between the two, compare the aligned phrases, and make a
    final entailment decision based on this aggregated comparison.  Each step in this process uses
    a feedforward network to modify the representation.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``headline`` and ``body`` ``TextFields`` we get as input to the
        model.
    attend_feedforward : ``FeedForward``
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the headline and words in the body.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between words in
        the headline and words in the body.
    compare_feedforward : ``FeedForward``
        This feedforward network is applied to the aligned headline and body representations,
        individually.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    headline_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the headline, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    body_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the body, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``headline_encoder`` for the encoding (doing nothing if ``headline_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 attend_feedforward: FeedForward,
                 similarity_function: SimilarityFunction,
                 compare_feedforward: FeedForward,
                 aggregate_feedforward: FeedForward,
                 headline_encoder: Optional[Seq2SeqEncoder] = None,
                 body_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DecomposableAttention, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward
        self._headline_encoder = headline_encoder
        self._body_encoder = body_encoder or headline_encoder

        self._num_stances = vocab.get_vocab_size(namespace="labels")

        check_dimensions_match(text_field_embedder.get_output_dim(), attend_feedforward.get_input_dim(),
                               "text field embedding dim", "attend feedforward input dim")
        check_dimensions_match(aggregate_feedforward.get_output_dim(), self._num_stances,
                               "final output dimension", "number of stances")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                headline: Dict[str, torch.LongTensor],
                body: Dict[str, torch.LongTensor],
                stance: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        headline : Dict[str, torch.LongTensor]
            From a ``TextField``
        body : Dict[str, torch.LongTensor]
            From a ``TextField``
        stance : torch.IntTensor, optional, (default = None)
            From a ``stanceField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the headline and
            body with 'headline_tokens' and 'body_tokens' keys respectively.
        Returns
        -------
        An output dictionary consisting of:

        stance_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_stances)`` representing unnormalised log
            probabilities of the entailment stance.
        stance_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_stances)`` representing probabilities of the
            entailment stance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
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
        stance_logits = self._aggregate_feedforward(aggregate_input)
        stance_probs = torch.nn.functional.softmax(stance_logits, dim=-1)

        output_dict = {"stance_logits": stance_logits,
                       "stance_probs": stance_probs,
                       "h2p_attention": h2p_attention,
                       "p2h_attention": p2h_attention}

        if stance is not None:
            loss = self._loss(stance_logits, stance.long().view(-1))
            self._accuracy(stance_logits, stance)
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["headline_tokens"] = [x["headline_tokens"] for x in metadata]
            output_dict["body_tokens"] = [x["body_tokens"] for x in metadata]

        return output_dict

    def decode(self, output_dict):
        class_probabilities = output_dict['stance_probs']

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        stance = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['stance'] = stance
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }
