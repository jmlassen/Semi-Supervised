import operator
from collections import Counter

import functools
from nltk.corpus import stopwords

DEFAULT_N_WORDS = 250
DEFAULT_SMOOTHING_VALUE = 1
# TODO: This should be number of unique words for target dictionary.


class SemiNbClassifier:
    def __init__(self, n_words=DEFAULT_N_WORDS, smoothing_value=DEFAULT_SMOOTHING_VALUE,
                stop=stopwords.words('english')):
        self.smoothing_value = smoothing_value
        self.stop = stop
        self.n_words = n_words
        self.word_counts = None
        self.target_count = None
        self.target_values = None

    def fit(self, data, target):
        self.target_values = self._get_unique_target_values(target)
        self.target_count = {}
        self._add_bags(self.target_values)
        for i, point in enumerate(data):
            bag = self._bag_point(point)
            self._add_word_counts(bag, target[i])
            self._add_target_count(target[i])

    def predict(self, data):
        results = []
        for point in data:
            probs = self._predict(point)
            largest_target = self._find_largest_target_prob(probs)
            results.append(largest_target)
        return results

    def _predict(self, point):
        probs = {}
        bag = self._bag_point(point)
        for target in self.target_values:
            probs[target] = self._calc_target_prob(bag, target)
        return probs

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

    def _add_word_counts(self, bag, target):
        for word in bag:
            if word[0] not in self.word_counts[target]:
                self.word_counts[target][word[0]] = 1
            else:
                self.word_counts[target][word[0]] += 1

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
