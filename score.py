import file_handling as fh

pred_file = 'preds/test_preds_decomp.txt'
test_file = 'data/test_data.csv'

pred_data = fh.read_jsonlist(pred_file)
test_data = fh.read_jsonlist(test_file)

n_total = 0.0
n_correct = 0.0

total_score = 0.0
top_score = 0.0
for pred_dict, gold_dict in zip(pred_data, test_data):
    pred = pred_dict['stance']
    gold = gold_dict['stance']
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
