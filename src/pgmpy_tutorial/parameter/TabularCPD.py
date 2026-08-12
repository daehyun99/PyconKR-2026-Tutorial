import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from pgmpy_tutorial.distributions.nominal import NominalDistribution
from pgmpy_tutorial.parameter._base import BaseParameter


class TabularCPD(BaseParameter):
    """
    Estimates a tabular conditional probability distribution for discrete variables.

    When fit without a target, this estimator learns the marginal distribution of
    a root variable. When fit with a target, it learns the conditional probability
    table of the target variable given the evidence variables.

    Parameters
    ----------
    categories: dict, optional
        Mapping from variable name to the discrete states that the variable can
        take. If unspecified, categories are inferred from the data passed to
        `fit`.

    evidences: dict, optional
        Mapping from evidence variable name to the discrete states that the
        evidence variable can take. If unspecified, evidence states are inferred
        from `X` when fitting a conditional distribution.

    Attributes
    ----------
    CPT_ : numpy.ndarray
        Learned conditional probability table. For a root variable, the table
        contains marginal probabilities. For a conditional distribution, rows
        correspond to target states and columns correspond to evidence-state
        configurations.

    columns_ : list
        Name of the variable whose distribution is represented by the learned
        table. Populated by `fit`.

    categories_ : dict
        Mapping from the target variable name to its learned or supplied states.
        Populated by `fit`.

    evidences_ : dict or None
        Mapping from evidence variable names to their learned or supplied states.
        Set to `None` for root-variable distributions. Populated by `fit`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.parameter.TabularCPD import TabularCPD
    >>> rng = np.random.default_rng(seed=42)
    >>> n_samples = 100
    >>> X = pd.DataFrame(
    ...     {
    ...         "x1": rng.integers(0, 3, size=n_samples),
    ...         "x2": rng.integers(0, 2, size=n_samples),
    ...     }
    ... )
    >>> y = pd.DataFrame({"y": rng.integers(0, 2, size=n_samples)})
    >>> cpd = TabularCPD()
    >>> cpd.fit(X, y)
    TabularCPD()
    >>> dist = cpd.predict_proba(X[:5])

    """

    _tags = {
        "variable_type": "discrete",
        "produces_factor": True,
        "is_linear_gaussian": False,
        "missing": False,
        "supports_fit_joint": False,
        "python_dependencies": ("skpro"),
    }

    def __init__(
        self,
        categories=None,
        evidences=None,
    ):
        self.categories = categories
        self.evidences = evidences
        super().__init__()

    def _fit(self, X, y=None, sample_weight=None):
        if y is None:
            # Unsupervised Learning
            if self.categories is None:
                self._y_transformer = LabelBinarizer()
                self._y_transformer.fit(X)
                self.categories_ = {X.columns[0]: self._y_transformer.classes_}
                self.evidences_ = self.evidences
            else:
                self.categories_ = self.categories
                self.evidences_ = self.evidences
        else:
            # Supervised Learning
            if self.categories is None:
                self._y_transformer = LabelBinarizer()
                self._y_transformer.fit(y)
                self.categories_ = {y.columns[0]: self._y_transformer.classes_}
            else:
                self.categories_ = self.categories

            if self.evidences is None:
                self._X_transformer = OneHotEncoder(
                    categories="auto",
                    handle_unknown="ignore",
                )
                self._X_transformer.fit(X)
                self.evidences_ = {
                    column: categories.tolist()
                    for column, categories in zip(
                        X.columns,
                        self._X_transformer.categories_,
                    )
                }
            else:
                self.evidences_ = self.evidences

        if y is None:
            # Unsupervised Learning: Root node
            weights = pd.Series(
                sample_weight,
                index=X.index,
            )
            counts = weights.groupby(
                [X[column] for column in list(X.columns)],
                observed=True,
                sort=True,
            ).sum()

            counts = counts.reindex(
                self.categories_[X.columns[0]],
                fill_value=0,
            )

            self.columns_ = [X.columns[0]]
            self.CPT_ = counts.div(counts.sum()).to_frame(name="prob")

        else:
            # Supervised Learning
            df = pd.concat([X, y], axis=1)

            evidence_names = list(X.columns)

            weights = pd.Series(
                sample_weight,
                index=df.index,
            )

            counts = weights.groupby(
                [df[column] for column in [y.columns[0], *evidence_names]],
                observed=True,
                sort=True,
                dropna=False,
            ).sum()

            counts = (
                counts.unstack(
                    evidence_names,
                    fill_value=0,
                )
                .reindex(
                    index=self.categories_[y.columns[0]],
                    fill_value=0,
                )
                .rename_axis(index=None)
            )

            self.columns_ = [y.columns[0]]
            self.CPT_ = counts.div(
                counts.sum(axis=0),
                axis=1,
            )

        self.CPT_ = np.asarray(self.CPT_)
        return self

    def _predict_proba(self, X):
        if self.evidences_ is None:
            # Unsupervised Learning
            probabilities = np.repeat(
                np.asarray(self.CPT_).T,
                repeats=len(X),
                axis=0,
            )

            return NominalDistribution(
                probs=probabilities,
                categories=self.categories_[self.columns_[0]],
                columns=self.columns_,
            )

        row_evidence = pd.MultiIndex.from_frame(X.loc[:, self.evidences_.keys()])
        cpt_column_index = pd.MultiIndex.from_product(
            [self.evidences_[name] for name in list(self.evidences_.keys())],
            names=list(self.evidences_.keys()),
        )
        column_positions = cpt_column_index.get_indexer(row_evidence)
        probabilities = self.CPT_[:, column_positions].T

        return NominalDistribution(
            probs=probabilities,
            categories=self.categories_[self.columns_[0]],
            columns=self.columns_,
        )  # (len(X), variable_card)

    def set_fitted_params(self, CPT, columns, categories, evidences, is_fitted):
        self.CPT_ = CPT
        self.columns_ = columns
        self.categories_ = categories
        self.evidences_ = evidences
        self._is_fitted = is_fitted
        return self
