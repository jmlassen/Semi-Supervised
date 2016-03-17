from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from multi_file_reader import MultiFileReader

DEFAULT_WORD_LENGTH = 2


class ClassifierFactory:
    def __init__(self, word_length=DEFAULT_WORD_LENGTH):
        self.word_length = word_length

    def build_benchmark_classifier(self, target_file_name, data_directory):
        print("Reading data...")
        tweet_train = MultiFileReader().read_labels_and_filenames(target_file_name, data_directory, self.word_length)
        classifier = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
                             ])
        # TODO: Remember you fool! there is no testing set when doing it this way!
        print("Joel, you should really look at your TODO comments...")
        print("Training classifier...")
        classifier.fit(tweet_train.data, tweet_train.target)
        return classifier

    def build_semi_supervised_classifier(self):
        pass