import csv
import os
import string


class TweetTrain(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target


class MultiFileReader:
    def read_labels_and_filenames(self, users_file_name, tweets_directory, word_length=0, remove_punct=True):
        data = []
        targets = []
        with open(users_file_name, encoding='utf8') as users_file:
            users_reader = csv.reader(users_file)
            for users_row in users_reader:
                user = users_row[0]
                target = users_row[1]
                tweets_file_name = "{}{}.csv".format(tweets_directory, user)
                if os.path.isfile(tweets_file_name) and (target == 'male' or target == 'female'):
                    # TODO: You should probably make the csv file encoding consistent, you fool
                    try:
                        tweets_file = open(tweets_file_name)
                        data.append(self._read_file(tweets_file, remove_punct, word_length))
                    except UnicodeDecodeError:
                        tweets_file = open(tweets_file_name, encoding='utf8')
                        data.append(self._read_file(tweets_file, remove_punct, word_length))
                    tweets_file.close()
                    targets.append(target)
        return TweetTrain(data, targets)

    def read_labeled_an_unlabeled_data(self, users_file_name, tweets_directory, word_length=0, remove_punct=True):
        data = []
        targets = []
        with open(users_file_name, encoding='utf8') as users_file:
            users_reader = csv.reader(users_file)
            for users_row in users_reader:
                user = users_row[0]
                target = users_row[1]
                tweets_file_name = "{}{}.csv".format(tweets_directory, user)
                # Here's where we change stuff, this needs to be combined
                if os.path.isfile(tweets_file_name) and target != 'remove':
                    try:
                        tweets_file = open(tweets_file_name)
                        data.append(self._read_file(tweets_file, remove_punct, word_length))
                    except UnicodeDecodeError:
                        tweets_file = open(tweets_file_name, encoding='utf8')
                        data.append(self._read_file(tweets_file, remove_punct, word_length))
                    tweets_file.close()
                    if (target == 'female') or (target == 'male'):
                        targets.append(target)
                    else:
                        # This could inconsistent, the value we assign needs to be tied to the value we set in
                        # semi_supervised_nb_classifier
                        targets.append(-1)
        return TweetTrain(data, targets)

    def _read_file(self, tweets_file, remove_punct, word_length):
        tweets_reader = csv.reader(tweets_file)
        tweets = ""
        for tweets_row in tweets_reader:
            if len(tweets_row) != 0:
                if remove_punct is True:
                    exclude = set(string.punctuation)
                    tweets_row[0] = ''.join(ch for ch in tweets_row[0] if ch not in exclude)
                if word_length > 0:
                    for word in tweets_row[0].split(' '):
                        if len(word) > word_length:
                            tweets += word.lower() + ' '
                else:
                    tweets += tweets_row[0] + ' '
        return tweets
