import file_handling as fh
import re
import pickle
import pandas as pd
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction import DictVectorizer

# (1) separately extract headlines and bodies; also trim the text
def extract_texts(data_list):
	headlines, bodies = [], []
	regex_line_break = re.compile('\\n{1,2}')
	regex_special = re.compile('[^\w\s\.,\-\*]+')
	for data in data_list:
		headlines.append(data['headline'])

		body = re.sub(regex_line_break, ' ', data['body'])
		body = re.sub(regex_special, '', body)
		bodies.append(body)

	return headlines, bodies

# (2) generate sentiment scores using SentimentIntensityAnalyzer, this generates
#     4 sentiment scores:  'neg', 'neu', 'pos', 'compound'. Each score ranges from 0 to 1    
def SentimentScoreFeature(text):
    sentiments = []
    sid = SentimentIntensityAnalyzer()
    for sentence in text:
        ss = sid.polarity_scores(sentence)
        sentiments.append(ss)
    # convert to panda dataframe
    senti_score = pd.DataFrame(sentiments)
    return senti_score


if __name__ == '__main__':

    train_data = 'fnc-1/train_data.csv'
    test_data = 'fnc-1/test_data.csv'
    dev_data  = 'fnc-1/dev_data.csv'

    train_data = fh.read_jsonlist(train_data)
    test_data = fh.read_jsonlist(test_data)
    dev_data = fh.read_jsonlist(dev_data)

    train_headlines, train_bodies = extract_texts(train_data)
    dev_headlines, dev_bodies = extract_texts(dev_data)
    test_headlines, test_bodies = extract_texts(test_data)

    train_headline_senti = SentimentScoreFeature(train_headlines)
    with open("train_headline_senti.pkl", "wb") as fp:   #Pickling
          pickle.dump(train_headline_senti, fp)

    train_bodies_senti = SentimentScoreFeature(train_bodies)
    with open("train_bodies_senti.pkl", "wb") as fp:   #Pickling
         pickle.dump(train_bodies_senti, fp)

    dev_headline_senti = SentimentScoreFeature(dev_headlines)
    with open("dev_headline_senti.pkl", "wb") as fp:   #Pickling
         pickle.dump(dev_headline_senti, fp)

    dev_bodies_senti = SentimentScoreFeature(dev_bodies)
    with open("dev_bodies_senti.pkl", "wb") as fp:   #Pickling
         pickle.dump(dev_bodies_senti, fp)

    test_headline_senti = SentimentScoreFeature(test_headlines)
    with open("test_headline_senti.pkl", "wb") as fp:   #Pickling
         pickle.dump(test_headline_senti, fp)

    test_bodies_senti = SentimentScoreFeature(test_bodies)
    with open("test_bodies_senti.pkl", "wb") as fp:   #Pickling
         pickle.dump(test_bodies_senti, fp)
