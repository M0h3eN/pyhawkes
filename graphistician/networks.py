import numpy as np

from pybasicbayes.abstractions import GibbsSampling

from graphistician.abstractions import NetworkDistribution, AdjacencyDistribution, WeightDistribution
import graphistician.adjacency as adjacency
import graphistician.weights as weights

class FactorizedNetworkDistribution(NetworkDistribution, GibbsSampling):
    def __init__(self, N,
                 adjacency_class, adjacency_kwargs,
                 weight_class, weight_kwargs):

        super(FactorizedNetworkDistribution, self).__init__()
        assert issubclass(adjacency_class, AdjacencyDistribution)
        self._adjacency = adjacency_class(N, **adjacency_kwargs)
        assert issubclass(weight_class, WeightDistribution)
        self._weights = weight_class(N, **weight_kwargs)

    @property
    def adjacency(self):
        return self._adjacency

    @property
    def weights(self):
        return self._weights

    def log_likelihood(self, A,W):
        lls = self.adjacency.log_likelihood(A)
        lls += self.weights.log_likelihood(A,W)
        return lls

    def log_prior(self):
        lp = 0
        lp += self.adjacency.log_prior()
        lp += self.weights.log_prior()
        return lp

    def rvs(self, *args, **kwargs):
        A = self.adjacency.rvs()
        W = self.weights.rvs()
        return A,W

    def sample_predictive_parameters(self):
        """
        Sample a predictive set of parameters for a new row and column of the network
        """
        Aprms = self.adjacency.sample_predictive_parameters()
        Wprms = self.adjacency.sample_predictive_parameters()
        return Aprms, Wprms

    def sample_predictive_distribution(self):
        """
        Sample a new row and column of the network
        """
        Arow, Acol = self.adjacency.sample_predictive_distribution()
        Wrow, Wcol = self.weights.sample_predictive_distribution()
        return Arow, Acol, Wrow, Wcol

    def approx_predictive_ll(self, Arow, Acol, Wrow, Wcol, M=100):
        """
        Approximate the predictive likelihood of a new row and column
        """
        pll = 0
        pll += self.adjacency.approx_predictive_ll(Arow, Acol, M=M)
        pll += self.weights.approx_predictive_ll(Arow, Acol, Wrow, Wcol, M=M)
        return pll

    def resample(self, A,W):
        self.adjacency.resample(A)
        self.weights.resample(A,W)

    def meanfieldupdate(self, expected_suff_stats):
        # TODO: Actually implement this!
        pass


class GaussianBernoulliNetwork(FactorizedNetworkDistribution):
    def __init__(self, N, B, p=0.5, mu=None, sigma=None):
        if mu is None:
            mu = np.zeros(B)
        if sigma is None:
            sigma = np.eye(B)

        super(GaussianBernoulliNetwork, self).__init__(N,
            adjacency.BernoulliAdjacencyDistribution, {"p": p},
            weights.FixedGaussianWeightDistribution, {"B": B, "mu": mu, "sigma": sigma}
        )


