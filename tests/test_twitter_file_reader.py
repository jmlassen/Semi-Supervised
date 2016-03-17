import unittest

from deprecated.twitter_file_reader import TwitterFileReader


class TestTwitterFileReader(unittest.TestCase):
    """Tests the TwitterFileReader class.

    """
    def test_read_users_file_and_tweets(self):
        """Tests the read_users_file_and_tweets method.

        :return: None
        """
        pass

    def test_get_labeled_users_list(self):
        """Tests the _get_labeled_users_list method.

        :return: None
        """
        tfr = TwitterFileReader()
        tfr._read_users_file_and_tweets('dataset/users.csv', 'dataset/tweets/')


if __name__ == '__main__':
    unittest.main()
