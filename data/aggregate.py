import pickle
import json
import numpy as np
import file_handling as fh
from tqdm import tqdm

def append_data(split_data, filepath, namespace):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    data = data.values.tolist()
    assert len(split_data) == len(data)

    for ex, d, in tqdm(zip(split_data, data), total=len(data)):
        ex[namespace] = d

def append_sims(split_data, filepath, namespace):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    data = [d.item() for d in data]
    assert len(split_data) == len(data)

    for ex, d, in tqdm(zip(split_data, data), total=len(data)):
        ex[namespace] = d

train_data = fh.read_jsonlist('train_data.csv')
dev_data = fh.read_jsonlist('dev_data.csv')
test_data = fh.read_jsonlist('test_data.csv')

append_data(train_data, 'train_bodies_senti.pkl', 'body_senti')
append_data(dev_data, 'dev_bodies_senti.pkl', 'body_senti')
append_data(test_data, 'test_bodies_senti.pkl', 'body_senti')


append_data(train_data, 'train_headline_senti.pkl', 'headline_senti')
append_data(dev_data, 'dev_headline_senti.pkl', 'headline_senti')
append_data(test_data, 'test_headline_senti.pkl', 'headline_senti')

append_sims(train_data, 'train_cos_sims.pkl', 'cos_sim')
append_sims(dev_data, 'dev_cos_sims.pkl', 'cos_sim')
append_sims(test_data, 'test_cos_sims.pkl', 'cos_sim')

fh.write_jsonlist(train_data, 'train_data.jsonl')
fh.write_jsonlist(dev_data, 'dev_data.jsonl')
fh.write_jsonlist(test_data, 'test_data.jsonl')
