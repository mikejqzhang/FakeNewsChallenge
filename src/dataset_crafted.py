import json

import logging

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ArrayField, LabelField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("fnc_dataset_crafted")
class FNCCraftedDataSetReader(DatasetReader):
    def __init__(self,
            tokenizer=None,
            token_indexers=None,
            lazy=False):
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path):
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                data_json = json.loads(line)
                headline = data_json['headline']
                body = data_json['body']
                stance = data_json['stance']
                crafted = data_json['crafted']
                yield self.text_to_instance(headline, body, crafted, stance)

    def text_to_instance(self, headline, body, crafted, stance=None):
        headline_tokens = self._tokenizer.tokenize(headline)
        body_tokens = self._tokenizer.tokenize(body)
        headline_field = TextField(headline_tokens, self._token_indexers)
        body_field = TextField(body_tokens, self._token_indexers)
        crafted_field = ArrayField(crafted)
        fields = {'headline': headline_field, 'body': body_field}
        if stance is not None:
            fields['stance'] = LabelField(stance)
        metadata = {"headline_tokens": [x.text for x in headline_tokens],
                    "body_tokens": [x.text for x in body_tokens]}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
