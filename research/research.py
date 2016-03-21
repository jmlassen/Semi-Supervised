from sklearn.utils import shuffle

from classifiers.semi_supervised_nb_classifier import SemiNbClassifier
from file_readers.multi_file_reader import MultiFileReader
from preprocessing.tweets_database import TweetsDatabase


def run_benchmark_test(tweets):
    results = []
    return results


def run_benchmark_graph_point_generation(tweets):
    # Set file names where we are going to save our results
    results_file = 'results-benchmark.csv'
    confusion_matrix_file = 'benchmark-confusion-matrix.csv'
    data, target = shuffle(tweets.data, tweets.target)
    for i in range(10, len(data[100:])):
        confusion_matrix = {}
        clf = SemiNbClassifier()
        clf.fit(data[100:(100 + i)], target[100:(100 + i)])
        predicted_values = clf.predict(data[:100])
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
        print("{} -> {}".format(i, accuracy))
        with open(results_file, "a") as file:
            file.write("{},{}\n".format(i, accuracy))
        print("{} -> {}\n".format(i, confusion_matrix))
        with open(confusion_matrix_file, "a") as file:
            file.write("{} -> {}\n".format(i, confusion_matrix))


def main():
    tweet_train = MultiFileReader().read_labeled_an_unlabeled_data('../dataset/users_new.csv', '../dataset/tweets/', 2)
    run_benchmark_graph_point_generation(tweet_train)

if __name__ == '__main__':
    main()
