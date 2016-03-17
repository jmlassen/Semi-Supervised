from preprocessing.tweets_database import TweetsDatabase


def run_benchmark_test(tweets):
    results = []

    return results


def main():
    tweets = TweetsDatabase().load_labeled_tweets()
    unlabeled_tweets = TweetsDatabase().load_unlabeled_tweets()
    benchmark_results = run_benchmark_test(tweets)


if __name__ == '__main__':
    main()
