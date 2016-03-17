import threading

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from classifiers.semi_supervised_nb_classifier import SemiNbClassifier
from utilities import cross_val_score


class Driver:
    def __init__(self):
        self.t_results = None

    def run_semi_test(self, tweet_train, critical_value):
        clf = SemiNbClassifier()
        results = cross_val_score(clf, tweet_train.data, tweet_train.target, critical_value)
        return np.mean(results), results

    def run_benchmark_test(self, tweet_train, critical_value):
        clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
        results = cross_val_score(clf, tweet_train.data, tweet_train.target, critical_value)
        return np.mean(results), results

    def run_small_semi_test(self):
        c = SemiNbClassifier()
        c.fit([
            "Inhabiting discretion the her dispatched decisively boisterous joy.",
            "So form were wish open is able of mile of. Waiting express if prevent it we an musical.",
            "Especially reasonable travelling she son. Resources resembled forfeited no to zealously.",
            "Has procured daughter how friendly followed repeated who surprise. Great asked oh under on voice downs.",
            "Law together prospect kindness securing six. Learning why get hastened smallest cheerful.",
            "Effect twenty indeed beyond for not had county. The use him without greatly can private."
        ], [
            "Male",
            "Male",
            "Male",
            "Female",
            "Female",
            "Female"
        ])
        results = c.predict([
            "His having within saw become ask passed misery giving. Recommend questions get too fulfilled. He fact in we case miss sake.",
            "Entrance be throwing he do blessing up. Hearts warmth in genius do garden advice mr it garret.",
            "Collected preserved are middleton dependent residence but him how. Handsome weddings yet mrs you has carriage packages.",
            "Preferred joy agreement put continual elsewhere delivered now.",
            "Mrs exercise felicity had men speaking met. Rich deal mrs part led pure will but."
        ])
        print(results)

    def run_graph_benchmark_classification(self, tweet_train):
        for i in range(302, len(tweet_train.target) - 50):
            results = []
            for _ in range(20):
                data, target = shuffle(tweet_train.data, tweet_train.target)
                clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
                clf.fit(data[:i], target[:i])
                prediction = clf.predict(data[(len(data) - 50):])
                results.append(accuracy_score(target[(len(data) - 50):], prediction))
                print("{} -> {}".format(i, results[-1]))
            with open("results-benchmark.csv", "a") as results_file:
                results_file.write("{},{}\n".format(i, round(np.mean(results), 3)))
            print("{} -> {}".format(i, np.mean(results)))

    def run_graph_benchmark_thread_classification(self, tweet_train):
        for i in range(313, len(tweet_train.target) - 50):
            threads = []
            self.t_results = []
            # create threads
            for _ in range(20):
                t = threading.Thread(target=self.run_one_class_test, args=[tweet_train, i, _])
                t.start()
                threads.append(t)
            print("Threads created")
            # join threads
            for j in range(20):
                threads[j].join()
            print("All threads joined. ({})".format(i))
            with open("results-benchmark.csv", "a") as results_file:
                results_file.write("{},{}\n".format(i, round(np.mean(self.t_results), 3)))
            print("Mean: {} -> {}".format(i, round(np.mean(self.t_results), 3)))

    def run_one_class_test(self, tweet_train, i, t):
        data, target = shuffle(tweet_train.data, tweet_train.target)
        clf = SemiNbClassifier()
        clf.fit(data[:i], target[:i])
        prediction = clf.predict(data[(len(data) - 50):])
        self.t_results.append(accuracy_score(target[(len(data) - 50):], prediction))
        print("({}){} -> {}".format(t, i, self.t_results[-1]))
