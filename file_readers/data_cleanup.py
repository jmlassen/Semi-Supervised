import csv
import os
import string
import webbrowser
from random import shuffle

from nltk.corpus import stopwords

users_file = '../dataset/users.csv'
tweets_dir = '../dataset/tweets/'
unlabeled_tweets_dir = '../dataset/unlabeled_tweets/'
new_users = '../dataset/users_new.csv'


def add_user_file(file, label):
    with open(new_users, 'a') as f:
        f.write("{},{}\n".format(file, label))


def print_gender_counts(records):
    genders = {}
    total = 0
    for record in records:
        if record[1] not in genders:
            genders[record[1]] = 1
        else:
            genders[record[1]] += 1
        if record[1] != "unlabeled" and not record[1].startswith("remove"):
            total += 1
    print("Total: {} -> {}".format(total, genders))


def possibly_non_english(filename):
    try:
        tweets_file = open(filename)
        tweets = _read_file(tweets_file)
    except UnicodeDecodeError:
        tweets_file = open(filename, encoding='utf8')
        tweets = _read_file(tweets_file)
    words = tweets.split(' ')
    stop_count = 0
    stop = stopwords.words('english')
    for word in words:
        if word in stop:
            stop_count += 1
    print("Stop count: {}".format(stop_count))
    if stop_count < 200:
        return True
    else:
        return False


def _read_file(file):
    tweets = ""
    file_reader = csv.reader(file)
    for row in file_reader:
        if len(row) > 0 and row[0].startswith("RT "):
            exclude = set(string.punctuation)
            row[0] = (''.join(ch for ch in row[0] if ch not in exclude)).lower()
            tweets += row[0] + ' '
    return tweets


def data_cleanup_ui():
    records = []
    # Get all the records in the file
    with open(new_users) as f:
        f_reader = csv.reader(f)
        for row in f_reader:
            records.append([row[0], row[1]])
    shuffle(records)
    for i in range(len(records)):
        url = "https://twitter.com/{}".format(records[i][0])
        filename = tweets_dir + records[i][0] + ".csv"
        if records[i][1] == 'unlabeled':
            print_gender_counts(records)
            webbrowser.open_new_tab(url)
            prompt_input = input("[1] M, [2] F, [3] R: ")
            if prompt_input == '1':
                records[i][1] = 'male'
            elif prompt_input == '2':
                records[i][1] = 'female'
            elif prompt_input == '3':
                why = input("Why: ")
                records[i][1] = 'remove-' + why
            elif prompt_input == 'q':
                break
            else:
                records[i][1] = "unlabeled-" + prompt_input
    os.remove(new_users)
    for record in records:
        add_user_file(record[0], record[1])


def main():
    with open(users_file, encoding='utf8') as user_f:
        if os.path.isfile(new_users):
            os.remove(new_users)
        users_reader = csv.reader(user_f)
        for users_row in users_reader:
            user = users_row[1]
            label = users_row[4]
            labeled_file = "{}{}.csv".format(tweets_dir, user)
            unlabeled_file = "{}{}.csv".format(unlabeled_tweets_dir, user)
            if label == 'Female' or label == 'Male':
                if os.path.isfile(labeled_file):
                    add_user_file(user, label.lower())
            elif label == 'UNLABELED':
                if os.path.isfile(unlabeled_file):
                    add_user_file(user, label.lower())


if __name__ == "__main__":
    # main()
    data_cleanup_ui()
