import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def precision_recall_fscore_support(y_true, y_pred, average='macro'):
    classes = np.unique(np.concatenate((y_true, y_pred))).tolist()
    precision_list, recall_list, fscore_list, support_list = [], [], [], []

    for cls in classes:
        true_pos = np.sum((y_true == cls) & (y_pred == cls))
        false_pos = np.sum((y_true != cls) & (y_pred == cls))
        false_neg = np.sum((y_true == cls) & (y_pred != cls))

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true == cls)

        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)
        support_list.append(support)

    macro_precision = np.mean(precision_list)
    macro_recall = np.mean(recall_list)
    macro_fscore = np.mean(fscore_list)
    macro_support = np.mean(support_list)

    return precision_list, recall_list, fscore_list, support_list, macro_precision, macro_recall, macro_fscore, macro_support



if __name__=='__main__':
  y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
  y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
  # print(y_true)
  result = precision_recall_fscore_support(y_true, y_pred, average='macro')
  print(result)
  acc = accuracy_score(y_true, y_pred)
  print(acc)