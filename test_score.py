import file_handling as fh

test_file = 'data/dev_data.csv'

test_data = fh.read_jsonlist(test_file)

n_total = 0.0
n_correct = 0.0

total_score = 0.0
top_score = 0.0
for gold_dict in test_data:
    gold = gold_dict['stance']
    if gold == 'unrelated':
        pred = 'unrelated'
    else:
        pred = 'discuss'

    n_total += 1
    n_correct += pred == gold
    
    if gold == 'unrelated':
        top_score += 0.25
        if pred == gold:
            total_score += 0.25
    else:
        top_score += 1.0
        if pred != 'unrelated':
            total_score += 0.25
        if pred == gold:
            total_score += 0.75

print('Accuracy: {} / {} | Percent: {}'.format(n_correct, n_total, n_correct / n_total))
print('Score: {} / {} | Normalized: {}'.format(total_score, top_score, total_score / top_score))
