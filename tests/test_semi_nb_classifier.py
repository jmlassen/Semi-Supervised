from unittest import TestCase

from classifiers.semi_supervised_nb_classifier import SemiNbClassifier


class TestSemiNbClassifier:
    def test_fit(self):
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


if __name__ == "__main__":
    t = TestSemiNbClassifier()
    t.test_fit()
