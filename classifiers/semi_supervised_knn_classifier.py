from collections import Counter

from nltk.corpus import stopwords

DEFAULT_N_MOST_COMMON_WORDS = 250


class SemiKnnClassifier:
    """Classifier for our semi supervised algorithm.

    """

    def __init__(self, k_neighbors=5, n_words=DEFAULT_N_MOST_COMMON_WORDS):
        self.points = []
        self.k_neighbors = k_neighbors
        self.stop = stopwords.words('english')
        self.n_words = n_words

    def fit(self, data, target, unlabeled_data=None):
        """Trains the classifier.

        :param unlabeled_data:
        :param data: Data containing the fields needed to train.
        :param target: The labels for the training data.
        :return: None

        """
        # Loop through 'labeled' data first.
        for i in range(len(data)):
            point = self._transform_data(data[i])
            self.points.append((point, target[i]))
        # Loop through each of our unlabeled training data, not required.
        if unlabeled_data is not None:
            for point in unlabeled_data:
                point = self._transform_data(point)
                self._add_unlabled_point(point)

    def predict(self, data):
        """Predicts which category the data falls into.

        :param data: Data containing the fields needed to predict (same as train).
        :return: Category that best fits based on the training data.
        """
        results = []
        for point in data:
            point = self._transform_data(point)
            results.append(self._predict(point))
        return results

    def _predict(self, point):
        """Private predict function that only predicts one point.

        :param point:
        :return:
        """
        neighbors = self._find_neighbors(point)
        return int(max(set(neighbors), key=neighbors.count))

    def _transform_data(self, data):
        bag = Counter()
        words = [j for j in data.split() if j not in self.stop]
        for word in words:
            if word != '':
                bag[word] += 1
        most_common_bag = Counter(bag.most_common(self.n_words))
        return most_common_bag

    def _add_unlabled_point(self, point):
        """Adds unlabeled points to our classifier.

        Finds the k nearest neighbors, if they are not all the same label...we throw out the record?

        :param point:
        :param stop:
        :return:
        """
        self.points.append((point, self._predict(point)))

    def _find_neighbors(self, data):
        """Finds the nearest neighbors.

        :return:
        """
        weights = []
        neighbors = []
        # Find the weight between each point
        for i in range(len(self.points)):
            weights.append((i, len(self.points[i][0] & data)))
        # Sort the weights
        weights.sort(key=lambda x: x[1], reverse=True)
        # Append the points to the neighbors list
        for weight in weights[:self.k_neighbors]:
            neighbors.append(self.points[weight[0]][1])
        return neighbors
