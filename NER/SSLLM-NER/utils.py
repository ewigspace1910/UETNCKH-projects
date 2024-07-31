def calculate_parameters(prediction, label):
    true_positive = sum([i == 1 and j == 1 for i, j in zip(prediction, label)])
    false_positive = sum([i == 1 and j == 0 for i, j in zip(prediction, label)])
    false_negative = sum([i == 0 and j == 1 for i, j in zip(prediction, label)])

    return true_positive, false_positive, false_negative

def calculate_precision(prediction, label):
    true_positive, false_positive, _ = calculate_parameters(prediction, label)
    
    if true_positive + false_positive == 0:
        return 0
    else:
        return true_positive / (true_positive + false_positive)
    
def calculate_recall(prediction, label):
    true_positive, _, false_negative = calculate_parameters(prediction, label)
    
    if true_positive + false_negative == 0:
        return 0
    else:
        return true_positive / (true_positive + false_negative)

def calculate_f1_score(prediction, label):
    precision = calculate_precision(prediction, label)
    recall = calculate_recall(prediction, label)
    
    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)