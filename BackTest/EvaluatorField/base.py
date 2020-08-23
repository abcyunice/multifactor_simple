'''
The base evaluator of all types

'''

from ._base import Evaluator


class FactorEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self._params = {"date", "stock", "return_rate", "factor", "field"}


class ReturnEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self._params = {"date", "stock", "return_rate", "weight", "field"}


class TurnoverEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self._params = {"date", "stock", "weight", "field"}


class EvaluatorSystem(Evaluator):
    def __init__(self):
        super().__init__()
        self._params = {"date", "stock", "return_rate", "factor", "field"}
