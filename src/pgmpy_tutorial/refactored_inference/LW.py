import networkx as nx
import numpy as np
import pandas as pd
from skpro.distributions import Empirical

from ._base import BaseInference


class LikelihoodWeighting(BaseInference):
    """Likelihood-weighting inference for Bayesian networks with fitted CPDs.

    The algorithm supports the parameter objects in :mod:`pgmpy.parameter`.
    It returns weighted samples instead of a discrete factor, which permits a
    single API for discrete and continuous variables.
    """

    _tags = {"individual": "BayesianNetwork"}

    def __init__(self):
        super().__init__()

    def query(self, model, variables, evidence=None, do=None, n_samples=10_000, seed=None):
        """Draw weighted posterior samples for ``variables``.

        Parameters
        ----------
        model : pgmpy.base._base._CoreGraph
            A directed acyclic graph with one fitted CPD per node, added with
            :meth:`~pgmpy.base._base._CoreGraph.add_cpd`.
        variables : str or list[str]
            Variables to retain in the returned samples.
        evidence : dict, optional
            Observed variable values. Their conditional probability (or
            density) is incorporated into each sample's importance weight.
        do : dict, optional
            Interventions mapping variables to fixed values. Intervened nodes
            are fixed without contributing a likelihood term.
        n_samples : int, default=10000
            Number of likelihood-weighted samples to generate.
        seed : int, numpy.random.Generator, optional
            Random state used for sampling.

        Returns
        -------
        skpro.distributions.Empirical
            Empirical distribution over ``variables`` whose samples are
            weighted by their normalized importance weights.
        """
        if isinstance(variables, str):
            variables = [variables]
        else:
            variables = list(variables)
        evidence = {} if evidence is None else dict(evidence)
        do = {} if do is None else dict(do)

        nodes = set(model.nodes)
        unknown = (set(variables) | set(evidence) | set(do)) - nodes
        if unknown:
            raise ValueError(f"Variables not found in the model: {unknown}")
        if set(evidence) & set(do):
            raise ValueError("A variable cannot be both evidence and an intervention.")
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")

        graph = model.get_directed_graph()
        if set(graph.edges) != {(u, v) for u, v, _ in model.get_edges(edge_types={"->"})}:
            raise ValueError("Likelihood weighting requires a model containing only directed edges.")
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Likelihood weighting requires an acyclic model.")

        order = list(nx.topological_sort(graph))
        missing_cpds = [node for node in order if node not in model.cpds]
        if missing_cpds:
            raise ValueError(f"The model has no CPDs for nodes: {missing_cpds}")

        rng = np.random.default_rng(seed)
        samples = []
        log_weights = np.empty(n_samples, dtype=float)
        for sample_index in range(n_samples):
            assignment = {}
            log_weight = 0.0
            for node in order:
                cpd = model.get_cpd(node)
                parents = list(cpd.evidences_) if getattr(cpd, "evidences_", None) is not None else []
                features = pd.DataFrame([{parent: assignment[parent] for parent in parents}])
                if not parents:
                    features = pd.DataFrame({node: [0]})
                distribution = cpd.predict_proba(features)
                if hasattr(distribution, "rng_"):
                    distribution.rng_ = rng

                if node in do:
                    assignment[node] = do[node]
                elif node in evidence:
                    assignment[node] = evidence[node]
                    values = pd.DataFrame({node: [assignment[node]]})
                    if cpd.get_tag("variable_type") == "discrete":
                        probability = distribution.pmf(values)
                    else:
                        probability = distribution.pdf(values)
                    with np.errstate(divide="ignore"):
                        log_weight += np.log(float(np.asarray(probability).reshape(-1)[0]))
                else:
                    assignment[node] = distribution.sample().iloc[0, 0]

            samples.append({variable: assignment[variable] for variable in variables})
            log_weights[sample_index] = log_weight

        max_log_weight = np.max(log_weights)
        if not np.isfinite(max_log_weight):
            raise ValueError("The evidence has zero probability under the model.")
        weights = np.exp(log_weights - max_log_weight)
        sample_index = pd.MultiIndex.from_product([[0], range(n_samples)], names=["instance", "sample"])
        result = pd.DataFrame(samples, index=sample_index)
        normalized_weights = pd.Series(weights / weights.sum(), index=sample_index)
        return Empirical(spl=result, weights=normalized_weights)
