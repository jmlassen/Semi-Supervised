from driver import Driver
from multi_file_reader import MultiFileReader

def main():
    tweet_train = MultiFileReader().read_labels_and_filenames('dataset/users_new.csv', 'dataset/tweets/', 2)
    driver = Driver()
    mean, results = driver.run_semi_test(tweet_train, 10)
    print("{} -> {}".format(mean, results))
    # mean, results = driver.run_benchmark_test(tweet_train, 10)
    # print("{} -> {}".format(mean, results))
    # driver.run_graph_benchmark_classification(tweet_train)
    # driver.run_graph_our_classification(tweet_train)
    # driver.run_graph_benchmark_thread_classification(tweet_train)

if __name__ == '__main__':
    main()
