import operator
from collections import Counter

import functools
from nltk.corpus import stopwords

DEFAULT_N_WORDS = 250
DEFAULT_SMOOTHING_VALUE = 1
DEFAULT_UNLABELED_VALUE = -1
DEFAULT_WEIGHT = 1


class SemiNbClassifier:
    def __init__(self, n_words=DEFAULT_N_WORDS, smoothing_value=DEFAULT_SMOOTHING_VALUE,
                 stop=stopwords.words('english'), unlabeled_value=DEFAULT_UNLABELED_VALUE, weight=DEFAULT_WEIGHT):
        self.n_words = n_words
        self.smoothing_value = smoothing_value
        self.stop = stop
        self.unlabeled_value = unlabeled_value
        self.weight = weight
        self.word_counts = None
        self.target_count = None
        self.target_values = None

    def fit(self, data, target):
        """

        Since the probability of an unlabeled point will change until all the labeled points are added to our
        classifier, we have to go through every record of the training data, and set unlabeled points aside until all
        the labeled points have been added. We then predict the classification of the unlabeled point before adding
        its words to our classifier.

        :param data:
        :param target:
        :return:
        """
        self.target_values = self._get_unique_target_values(target)
        self.target_count = {}
        self._add_bags(self.target_values)
        unlabeled_bags = []
        for i, point in enumerate(data):
            bag = self._bag_point(point)
            if target[i] == self.unlabeled_value:
                unlabeled_bags.append(bag)
            self._add_word_counts(bag, target[i], self.weight)
            self._add_target_count(target[i])
        for unlabeled_bag in unlabeled_bags:
            probabilities = self._predict(unlabeled_bag)
            print(probabilities)

    def predict(self, data):
        results = []
        for point in data:
            bag = self._bag_point(point)
            probabilities = self._predict(bag)
            largest_target = self._find_largest_target_prob(probabilities)
            results.append(largest_target)
        return results

    def _predict(self, bag):
        probabilities = {}
        for target in self.target_values:
            probabilities[target] = self._calc_target_prob(bag, target)
        return probabilities

    def _bag_point(self, point):
        bag = Counter()
        words = [j for j in point.split() if j not in self.stop]
        for word in words:
            if word != '':
                bag[word] += 1
        most_common_bag = Counter(bag.most_common(self.n_words))
        return most_common_bag

    def _add_bags(self, target_values):
        self.word_counts = {}
        for value in target_values:
            self.word_counts[value] = {}

    def _get_unique_target_values(self, target):
        return set(target)

    def _find_largest_target_prob(self, probs):
        return max(probs, key=probs.get)

    def _add_word_counts(self, bag, target, weight):
        for word in bag:
            if word[0] not in self.word_counts[target]:
                self.word_counts[target][word[0]] = weight
            else:
                self.word_counts[target][word[0]] += weight

    def _calc_target_prob(self, bag, target):
        probs = [self._get_target_prob(target)]
        for word in bag:
            if word[0] not in self.word_counts[target]:
                value = self.smoothing_value
            else:
                value = self.word_counts[target][word[0]] + self.smoothing_value
                # print("'{}' found with count {} for target {}".format(word[0], value, target))
            probs.append(value / (self.target_count[target] + len(self.word_counts)))
            # print("Probability found to be {}".format(probs[-1]))
        return functools.reduce(operator.mul, probs, 1)

    def _add_target_count(self, target):
        if target not in self.target_count:
            self.target_count[target] = 1
        else:
            self.target_count[target] += 1

    def _get_target_prob(self, target):
        sum = 0
        for item in self.target_count:
            sum += self.target_count[item]
        return self.target_count[target] / sum
