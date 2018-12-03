from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import file_handling as fh
import re
import pickle
from tqdm import tqdm

# (1)preprocess the text and separately extract headlines, bodies, stances, and pairs of headlines+bodies
def extract_all(data_list):
    headlines, bodies, stances, pairs = [], [], [], []
    regex_line_break = re.compile('\\n{1,2}')
    regex_special = re.compile('[^\w\s\.,\-\*]+')
    for data in tqdm(data_list):
        headlines.append(data['headline'])
        stances.append(data['stance'])

        body = re.sub(regex_line_break, ' ', data['body'])
        body = re.sub(regex_special, '', body)
        bodies.append(body)

        pairs.append(' '.join([data['headline'], body]))

        pair_list = list(zip(headlines, bodies))
    return headlines, bodies, stances, pairs, pair_list

# (2)generate Tf_idf vectors using ngram = 1 or 2 or 3
def TfidfGenerator(data, ngram):
    #create the ngram transform
    vectorizer = TfidfVectorizer(ngram_range=(1, ngram))
    # tokenize and build vocabulary
    vectorizer.fit(data)
    # encode tfidf to data
    tfidf = vectorizer.transform(data)
    return tfidf

# (3)compute cosine similarity between headline tfidf features and body tfidf features
def calculate_cos_sim(vec1, vec2):
    cos = cosine_similarity(vec1, vec2)
    cos_sim = np.asarray(cos)
    return cos_sim

# (4*) compute the tfidf for each pair of headline and body and calculate the cosine similarity (within one instance)
def TfidfcossimFeature(data):
    cos_sims = []
    for pair in tqdm(data):
        vec = TfidfGenerator(pair, ngram=1)
        cos_sim = calculate_cos_sim(vec[0], vec[1])
        cos_sims.append(cos_sim)
    return cos_sims

if __name__ == '__main__':
    train_data = 'data/train_data.csv'
    test_data = 'data/test_data.csv'
    dev_data  = 'data/dev_data.csv'

    train_data = fh.read_jsonlist(train_data)
    test_data = fh.read_jsonlist(test_data)
    dev_data = fh.read_jsonlist(dev_data)

    train_headlines, train_bodies, train_stances, train_pairs, train_pair_list = extract_all(train_data)
    dev_headlines, dev_bodies, dev_stances, dev_pairs, dev_pair_list = extract_all(dev_data)
    test_headlines, test_bodies, test_stances, test_pairs, test_pair_list = extract_all(test_data)


    train_cos_sim = TfidfcossimFeature(train_pair_list)
    with open("train_cos_sims.pkl", "wb") as fp:   #Pickling
        pickle.dump(train_cos_sim, fp)

    dev_cos_sim = TfidfcossimFeature(dev_pair_list)
    with open("dev_cos_sims.pkl", "wb") as fp:   #Pickling
        pickle.dump(dev_cos_sim, fp)

    test_cos_sim = TfidfcossimFeature(test_pair_list)
    with open("test_cos_sims.pkl", "wb") as fp:   #Pickling
        pickle.dump(test_cos_sim, fp)
