from abc import abstractmethod
from sklearn.metrics import accuracy_score


class Evaluator(object):

    @property
    @abstractmethod
    def worst_score(self):
        pass

    @property
    @abstractmethod
    def mode(self):
        pass

    @abstractmethod
    def score(self, y_true, y_pred, **kwargs):
        pass

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        pass


class AccuracyEvaluator(Evaluator):

    @property
    def worst_score(self):
        return 0.0

    @property
    def mode(self):
        return 'maximize'

    def score(self, y_true, y_pred):
        return accuracy_score(y_true=y_true.argmax(axis=1), y_pred=y_pred.argmax(axis=1))

    def is_better(self, curr, best, **kwargs):
        eps = kwargs.pop('accuracy_eps', 1e-4)
        return curr > best * (1. + eps)

