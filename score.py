import file_handling as fh

pred_file = 'test_preds.txt'
test_file = 'fnc-1/test_data.csv'

pred_data = fh.read_jsonlist(pred_file)
test_data = fh.read_jsonlist(test_file)

total_score = 0.0
top_score = 0.0
for pred_dict, gold_dict in zip(pred_data, test_data):
    pred = pred_dict['stance']
    gold = gold_dict['stance']
    
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

print('Score: {} / {} | Normalized: {}'.format(total_score, top_score, total_score / top_score))
