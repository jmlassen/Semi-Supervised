import csv
import os
import webbrowser

users_file = '../dataset/users.csv'
tweets_dir = '../dataset/tweets/'
unlabeled_tweets_dir = '../dataset/unlabeled_tweets/'
new_users = '../dataset/users_new.csv'


def add_user_file(file, label):
    with open(new_users, 'a') as f:
        f.write("{},{}\n".format(file, label))


def print_gender_counts(records):
    genders = {}
    for record in records:
        if record[1] not in genders:
            genders[record[1]] = 1
        else:
            genders[record[1]] += 1
    print(genders)


def main_new():
    records = []
    # Get all the records in the file
    with open(new_users) as f:
        f_reader = csv.reader(f)
        for row in f_reader:
            records.append([row[0], row[1]])
    for i in range(len(records)):
        if records[i][1] != 'remove' and records[i][1] != 'unlabeled-l':
            print_gender_counts(records)
            url = "https://twitter.com/{}".format(records[i][0])
            webbrowser.open_new_tab(url)
            gender = input("[1] M, [2] F, [3] U, [4] R: ")
            if gender == '4':
                print("Removing {}.csv".format(records[i][0]))
                records[i][1] = 'remove'
            elif gender == '1':
                records[i][1] = 'male'
            elif gender == '2':
                records[i][1] = 'female'
            elif gender == '3':
                records[i][1] = 'unlabeled-l'
            else:
                break
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
    main_new()
