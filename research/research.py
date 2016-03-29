from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from classifiers.semi_supervised_knn_classifier import SemiKnnClassifier
from classifiers.semi_supervised_nb_classifier import SemiNbClassifier
from file_readers.multi_file_reader import MultiFileReader


def run_benchmark_test(tweets):
    results = []
    return results


def run_benchmark_graph_point_generation(tweets, run_with_unlabeled=False):
    # Set file names where we are going to save our results
    results_file = 'semi-nb-results-low-unlabeled-weight.csv'
    confusion_matrix_file = 'semi-nb-confusion-matrix-low-unlabeled-weight.csv'
    data, target = shuffle(tweets.data, tweets.target)
    clf = SemiNbClassifier()
    clf_not_semi = SemiNbClassifier()
    for i in range(10, len(data[100:])):
        confusion_matrix = {}
        clf.fit(data[100:(100 + i)], target[100:(100 + i)], tweets.unlabeled)
        clf_not_semi.fit(data[100:(100 + i)], target[100:(100 + i)])
        predicted_values = clf.predict(data[:100])
        predicted_values_not_semi = clf_not_semi.predict(data[:100])
        correct = 0
        for j, prediction_point in enumerate(predicted_values):
            if prediction_point == target[j]:
                correct += 1
            # Voodoo for creating confusion matrix
            if target[j] not in confusion_matrix:
                confusion_matrix[target[j]] = {}
            if prediction_point not in confusion_matrix[target[j]]:
                confusion_matrix[target[j]][prediction_point] = 1
            else:
                confusion_matrix[target[j]][prediction_point] += 1
        accuracy = round(correct / len(data[:100]), 5)
        accuracy_not_semi = accuracy_score(target[:100], predicted_values_not_semi)
        print("(No Labeled Data) {} -> {}".format(i, accuracy_not_semi))
        print("{} -> {}".format(i, accuracy))
        with open(results_file, "a") as file:
            file.write("{},{}\n".format(i, accuracy))
        print("{} -> {}".format(i, confusion_matrix))
        with open(confusion_matrix_file, "a") as file:
            for target_val in confusion_matrix:
                if target_val in confusion_matrix[target_val]:
                    file.write("{},".format(confusion_matrix[target_val][target_val]))
                else:
                    file.write("{},".format(0))
            file.write("{} -> {}\n".format(i, confusion_matrix))


def main():
    tweet_train = MultiFileReader().read_labeled_an_unlabeled_data('../dataset/users_new.csv', '../dataset/tweets/', 2)
    run_benchmark_graph_point_generation(tweet_train, run_with_unlabeled=True)

if __name__ == '__main__':
    main()
