from skpro.regression.base._delegate import _DelegatedProbaRegressor

from pgmpy_tutorial.parameter._base import BaseParameter


class SkproAdapter(_DelegatedProbaRegressor, BaseParameter):
    _tags = {
        "parameter_type": "regressor",
        "produces_factor": False,
        "is_linear_gaussian": False,
        "supports_fit_joint": False,
        "missing": False,
        "can_be_root": False,
    }
    _delegate_name = "estimator"

    def __init__(self, estimator):
        self.estimator = estimator
        super().__init__()

    def _fit(self, X, y, sample_weight=None):

        estimator = self._get_delegate()
        self.evidences_ = list(X.columns)
        estimator.fit(X=X, y=y, C=sample_weight)
        return self

    def _predict_proba(self, X):
        estimator = self._get_delegate()
        return estimator.predict_proba(X=X)
