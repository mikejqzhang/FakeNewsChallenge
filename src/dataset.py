import json

import logging

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("fnc_dataset")
class FNCDataSetReader(DatasetReader):
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
                yield self.text_to_instance(headline, body, stance)

    def text_to_instance(self, headline, body, stance=None):
        tokenized_headline = self._tokenizer.tokenize(headline)
        tokenized_body = self._tokenizer.tokenize(body)
        headline_field = TextField(tokenized_headline, self._token_indexers)
        body_field = TextField(tokenized_body, self._token_indexers)
        fields = {'headline': headline_field, 'body': body_field}
        if stance is not None:
            fields['stance'] = LabelField(stance)
        return Instance(fields)
