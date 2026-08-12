import numpy as np
from sklearn.base import is_classifier, is_regressor
from skpro.regression.base._delegate import _DelegatedProbaRegressor

from pgmpy_tutorial.parameter._base import BaseParameter


class SklearnAdapter(_DelegatedProbaRegressor, BaseParameter):
    _tags = {
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
        if is_regressor(estimator):
            self.set_tags(variable_type="continuous")
        elif is_classifier(estimator):
            if not callable(getattr(estimator, "predict_proba", None)):
                raise TypeError(
                    "Currently, only classifier models that implement predict_proba() "
                    "are supported. "
                    f"Received: {type(estimator).__name__}"
                )
            self.set_tags(variable_type="discrete")
        else:
            raise TypeError(
                f"estimator must be a scikit-learn classifier or regressor. Received: {type(estimator).__name__}"
            )

    def _fit(self, X, y, sample_weight=None):

        estimator = self._get_delegate()
        self.columns_ = y.columns[0]
        self.evidences_ = list(X.columns)
        estimator.fit(X=X, y=y, sample_weight=sample_weight)

        if is_regressor(estimator):
            y_arr = y.to_numpy(dtype=float).reshape(-1)
            residuals = y_arr - estimator.predict(X).reshape(-1)

            if sample_weight is None:
                sample_weight = np.ones(len(y), dtype=float)
            else:
                sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)

            self.std_ = np.sqrt(np.sum(sample_weight * residuals**2) / (np.sum(sample_weight)))

        return self

    def _predict_proba(self, X):
        estimator = self._get_delegate()

        if is_regressor(estimator):
            from skpro.distributions.normal import Normal

            mu = estimator.predict(X)
            sigma = self.std_
            return Normal(mu, sigma, index=X.index, columns=[self.columns_])

        elif is_classifier(estimator):
            from pgmpy_tutorial.distributions.nominal import NominalDistribution

            probs = estimator.predict_proba(X)
            return NominalDistribution(
                probs=probs, categories=estimator.classes_, index=X.index, columns=[self.columns_]
            )
