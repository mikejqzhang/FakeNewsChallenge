from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import file_handling as fh
import re
import pickle

# (1)preprocess the text and separately extract headlines, bodies, stances, and pairs of headlines+bodies
def extract_all(data_list):
	headlines, bodies, stances, pairs = [], [], [], []
	regex_line_break = re.compile('\\n{1,2}')
	regex_special = re.compile('[^\w\s\.,\-\*]+')
	for data in data_list:
		headlines.append(data['headline'])
		stances.append(data['stance'])

		body = re.sub(regex_line_break, ' ', data['body'])
		body = re.sub(regex_special, '', body)
		bodies.append(body)

		pairs.append(' '.join([data['headline'], body]))

		pair_list = list(zip(headlines, bodies))
	return headlines, bodies, stances, pairs, pair_list

if __name__ == '__main__':

    train_data = 'fnc-1/train_data.csv'
    test_data = 'fnc-1/test_data.csv'
    dev_data  = 'fnc-1/dev_data.csv'

    train_data = fh.read_jsonlist(train_data)
    test_data = fh.read_jsonlist(test_data)
    dev_data = fh.read_jsonlist(dev_data)

    # train_headlines, train_bodies, train_stances, train_pairs, train_pair_list = extract_all(train_data)
    dev_headlines, dev_bodies, dev_stances, dev_pairs, dev_pair_list = extract_all(dev_data)
    # test_headlines, test_bodies, test_stances, test_pairs, test_pair_list = extract_all(test_data)

    def MatchedWordFeature(pair_list, ngram):
        m = []
        for pair in pair_list:
            headline = pair[0]
            body = pair[1]
            vec_body = CountVectorizer([body], ngram_range=(1, ngram))
            vec_body.fit([body])

            vec = CountVectorizer([headline], ngram_range=(1, ngram))
            vec.fit([headline])
            vec_headline = vec.transform([headline])
            vec_headline.toarray()
            # print(vec_headline.shape[1]) # this counts how many words there is in the headline

            headline_transform = vec_body.transform([headline])
            headline_transform.toarray()
            # count how many words in headline appears in the body, and normalize by the length of the headline
            matched = np.count_nonzero(headline_transform.toarray())/vec_headline.shape[1]
            m.append(matched)

        return m


    train_matched_ratio = MatchedWordFeature(train_pair_list, 1)
    dev_matched_ratio = MatchedWordFeature(dev_pair_list, 1)
    test_matched_ratio = MatchedWordFeature(test_pair_list, 1)
