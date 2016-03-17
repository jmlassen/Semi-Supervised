from sklearn.utils import shuffle
import numpy as np


def cross_val_score(classifier, data, target, critical_value):
    data, target = shuffle(data, target)
    fold_len = int(len(data) / critical_value)
    results = []
    for i in range(critical_value):
        start_index, end_index = _calc_start_and_end_indexes(fold_len, i)
        train_data = np.concatenate((data[:start_index], data[end_index:]), axis=0)
        train_target = np.concatenate((target[:start_index], target[end_index:]), axis=0)
        classifier.fit(train_data, train_target)
        predictions = classifier.predict(data[start_index:end_index])
        correct_predictions = 0
        for i in range(len(predictions)):
            if predictions[i] == target[start_index + i]:
                correct_predictions += 1
        accuracy = correct_predictions / len(predictions)
        results.append(accuracy)
    return np.array(results)


def _calc_start_and_end_indexes(fold_len, index):
    start_index = fold_len * index
    end_index = fold_len * (index + 1)
    return start_index, end_index
