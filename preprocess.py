import csv
import random
import file_handling as fh

def merge_data(stances_file, bodies_file):
    with open(stances_file, 'r') as stances_f:
        stances = list(csv.DictReader(stances_f))
    with open(bodies_file, 'r') as bodies_f:
        bodies = dict(csv.reader(bodies_f))

    data = [{'headline': s['Headline'],
            'stance': s['Stance'],
            'body_id': s['Body ID'],
            'body': bodies[s['Body ID']]
            } for s in stances]
    return data

if __name__ == '__main__':
    random.seed(3000)
    split = 0.8

    train_stances = 'fnc-1/train_stances.csv'
    train_bodies = 'fnc-1/train_bodies.csv' 
    test_stances = 'fnc-1/competition_test_stances.csv'
    test_bodies = 'fnc-1/competition_test_bodies.csv'

    train_data = 'fnc-1/train_data.csv' 
    dev_data = 'fnc-1/dev_data.csv' 
    test_data = 'fnc-1/test_data.csv'

    train = merge_data(train_stances, train_bodies)
    random.shuffle(train)
    split_ind = int(len(train) * split)

    train, dev = train[:split_ind], train[split_ind:]
    test = merge_data(test_stances, test_bodies)

    fh.write_jsonlist(train, train_data, sort_keys=True)
    fh.write_jsonlist(dev, dev_data, sort_keys=True)
    fh.write_jsonlist(test, test_data, sort_keys=True)
