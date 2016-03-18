from unittest import TestCase

from classifiers.semi_supervised_nb_classifier import SemiNbClassifier


class TestSemiNbClassifier:
    def test_fit(self):
        c = SemiNbClassifier()
        c.fit([
            "Inhabiting discretion the her dispatched decisively boisterous joy".lower(),
            "So form were wish open is able of mile of Waiting express if prevent it we an musical".lower(),
            "Especially reasonable travelling she son Resources resembled forfeited no to zealously".lower(),
            "Has procured daughter how friendly followed repeated who surprise Great asked oh under on voice".lower(),
            "Law together prospect kindness securing six Learning why get hastened smallest cheerful".lower(),
            "Effect twenty indeed beyond for not had county The use him without greatly can private.".lower(),
            "His having within saw become ask passed misery giving Recommend questions get too fulfilled".lower()
        ], [
            "Male",
            "Male",
            "Male",
            "Female",
            "Female",
            "Female",
            "-1"
        ])
        return


if __name__ == "__main__":
    t = TestSemiNbClassifier()
    t.test_fit()
