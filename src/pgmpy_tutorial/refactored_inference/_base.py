from skbase.base import BaseEstimator as _BaseEstimator


class BaseInference(_BaseEstimator):
    _tags = {
        "individual": str,
    }

    def __init__(self):
        super().__init__()

    def query(self, model, variables, evidence=None, do=None):
        raise NotImplementedError
