"""
Top level classes for the Hawkes process model.
"""
import abc
import copy

import numpy as np
from scipy.special import gammaln

from pyhawkes.deps.pybasicbayes.models import ModelGibbsSampling, ModelMeanField
from pyhawkes.internals.bias import GammaBias
from pyhawkes.internals.weights import SpikeAndSlabGammaWeights, GammaMixtureWeights
from pyhawkes.internals.impulses import DirichletImpulseResponses
from pyhawkes.internals.parents import Parents
from pyhawkes.internals.network import ErdosRenyiModel
from pyhawkes.utils.basis import CosineBasis

class _DiscreteTimeNetworkHawkesModelBase(object):
    """
    Discrete time network Hawkes process model with support for
    Gibbs sampling inference, variational inference (TODO), and
    stochastic variational inference (TODO).
    """

    __metaclass__ = abc.ABCMeta
    _weight_class = None
    _parent_class = None

    def __init__(self, K, dt=1.0, dt_max=10.0,
                 B=5, basis=None,
                 alpha0=1.0, beta0=1.0,
                 kappa=1.0, v=5.0, p=0.9,
                 gamma=1.0):
        """
        Initialize a discrete time network Hawkes model with K processes.

        :param K:  Number of processes
        """
        self.K = K
        self.dt = dt
        self.dt_max = dt_max

        # Initialize the basis
        self.B = B
        self.basis = CosineBasis(self.B, self.dt, self.dt_max, norm=True)

        # Initialize the model components
        self.bias_model = GammaBias(self.K, self.dt, alpha0, beta0)
        self.impulse_model = DirichletImpulseResponses(self.K, self.B, gamma=gamma)

        # Initialize the network model
        self.network = ErdosRenyiModel(self.K, p=p, kappa=kappa, v=v)

        # The weight model is dictated by whether this is for Gibbs or MF
        self.weight_model = self._weight_class(self.K, self.network)

        # Initialize the data list to empty
        self.data_list = []

    def add_data(self, S):
        """
        Add a data set to the list of observations.
        First, filter the data with the impulse response basis,
        then instantiate a set of parents for this data set.

        :param S: a TxK matrix of of event counts for each time bin
                  and each process.
        """
        assert isinstance(S, np.ndarray) and S.ndim == 2 and S.shape[1] == self.K \
               and np.amin(S) >= 0 and S.dtype == np.int, \
               "Data must be a TxK array of event counts"

        T = S.shape[0]
        N = np.atleast_1d(S.sum(axis=0))

        # Filter the data into a TxKxB array
        F = self.basis.convolve_with_basis(S)

        # # Check that \sum_t F[t,k,b] ~= Nk / dt
        # Fsum = F.sum(axis=0)
        # print "F_err:  ", Fsum - N/self.dt

        # Instantiate corresponding parent object
        parents = self._parent_class(T, self.K, self.B, S, F)
        parents.resample(self.bias_model, self.weight_model, self.impulse_model)

        # Add to the data list
        self.data_list.append((S, N, F, parents))

    def check_stability(self):
        """
        Check that the weight matrix is stable

        :return:
        """
        eigs = np.linalg.eigvals(self.weight_model.A * self.weight_model.W)
        maxeig = np.amax(np.real(eigs))
        # print "Max eigenvalue: ", maxeig
        if maxeig < 1.0:
            return True
        else:
            return False

    def generate(self, keep=True, T=100):
        """
        Generate a new data set with the sampled parameters

        :param keep: If True, add the generated data to the data list.
        :param T:    Number of time bins to simulate.
        :return: A TxK
        """
        assert isinstance(T, int), "T must be an integer number of time bins"

        # Test stability
        self.check_stability()

        # Initialize the output
        S = np.zeros((T, self.K))

        # Precompute the impulse responses (LxKxK array)
        G = np.tensordot(self.basis.basis, self.impulse_model.g, axes=([1], [2]))
        L = self.basis.L
        assert G.shape == (L,self.K, self.K)
        H = self.weight_model.A[None,:,:] * \
            self.weight_model.W[None,:,:] * \
            G

        # Compute the rate matrix R
        R = np.zeros((T+L, self.K))

        # Add the background rate
        R += self.bias_model.lambda0[None,:]

        # Iterate over time bins
        for t in xrange(T):
            # Sample a Poisson number of events for each process
            S[t,:] = np.random.poisson(R[t,:] * self.dt)

            # For each sampled event, add a weighted impulse response to the rate
            for k in xrange(self.K):
                if S[t,k] > 0:
                    R[t+1:t+L+1,:] += S[t,k] * H[:,k,:]

            # Check Spike limit
            if np.any(S[t,:] >= 1000):
                print "More than 1000 events in one time bin!"
                import pdb; pdb.set_trace()

        # Only keep the first T time bins
        S = S[:T,:].astype(np.int)
        R = R[:T,:]

        if keep:
            # Xs = [X[:T,:] for X in Xs]
            # data = np.hstack(Xs + [S])
            self.add_data(S)


        return S, R

    def get_parameters(self):
        """
        Get a copy of the parameters of the model
        :return:
        """
        return self.weight_model.A, \
               self.weight_model.W, \
               self.impulse_model.g, \
               self.bias_model.lambda0

    def set_parameters(self, params):
        """
        Set the parameters of the model
        :param params:
        :return:
        """
        A, W, beta, lambda0 = params
        K, B = self.K, self.basis.B

        assert isinstance(A, np.ndarray) and A.shape == (K,K), \
            "A must be a KxK adjacency matrix"

        assert isinstance(W, np.ndarray) and W.shape == (K,K) \
               and np.amin(W) >= 0, \
            "W must be a KxK weight matrix"

        assert isinstance(beta, np.ndarray) and beta.shape == (K,K,B) and \
               np.allclose(beta.sum(axis=2), 1.0), \
            "beta must be a KxKxB impulse response array"

        assert isinstance(lambda0, np.ndarray) and lambda0.shape == (K,) \
               and np.amin(lambda0) >=0, \
            "lambda0 must be a K-vector of background rates"

        self.weight_model.A = A
        self.weight_model.W = W
        self.impulse_model.g = beta
        self.bias_model.lambda0 = lambda0

    def compute_rate(self, index=0, proc=None, S=None):
        """
        Compute the rate function for a given data set
        :param index:   An integer specifying which dataset (if S is None)
        :param S:       TxK array of event counts for which we would like to
                        compute the model's rate
        :return:        TxK array of rates
        """
        if S is not None:
            assert isinstance(S, np.ndarray) and S.ndim == 2, "S must be a TxK array."
            T,K = S.shape

            # Filter the data into a TxKxB array
            F = self.basis.convolve_with_basis(S)

        else:
            assert len(self.data_list) > index, "Dataset %d does not exist!" % index
            S, _, F, _ = self.data_list[index]
            T,K = S.shape

        if proc is None:
            # Compute the rate
            R = np.zeros((T,K))

            # Background rate
            R += self.bias_model.lambda0[None,:]

            # Compute the sum of weighted sum of impulse responses
            H = self.weight_model.A[:,:,None] * \
                self.weight_model.W[:,:,None] * \
                self.impulse_model.g

            H = np.transpose(H, [2,0,1])

            for k2 in xrange(self.K):
                R[:,k2] += np.tensordot(F, H[:,:,k2], axes=([2,1], [0,1]))

            return R

        else:
            assert isinstance(proc, int) and proc < self.K, "Proc must be an int"
            # Compute the rate
            R = np.zeros((T,))

            # Background rate
            R += self.bias_model.lambda0[proc]

            # Compute the sum of weighted sum of impulse responses
            H = self.weight_model.A[:,proc,None] * \
                self.weight_model.W[:,proc,None] * \
                self.impulse_model.g[:,proc,:]

            R += np.tensordot(F, H, axes=([1,2], [0,1]))

            return R

    def _poisson_log_likelihood(self, S, R):
        """
        Compute the log likelihood of a Poisson matrix with rates R

        :param S:   Count matrix
        :param R:   Rate matrix
        :return:    log likelihood
        """
        return (-gammaln(S+1) + S * np.log(R*self.dt) - R*self.dt).sum()

    def heldout_log_likelihood(self, S):
        """
        Compute the held out log likelihood of a data matrix S.
        :param S:   TxK matrix of event counts
        :return:    log likelihood of those counts under the current model
        """
        R = self.compute_rate(S=S)
        return self._poisson_log_likelihood(S, R)

    def log_likelihood(self):
        """
        Compute the joint log probability of the data and the parameters
        :return:
        """
        ll = 0

        # Get the likelihood of the datasets
        for ind,(S,_,_,_)  in enumerate(self.data_list):
            R = self.compute_rate(index=ind)
            ll += self._poisson_log_likelihood(S,R)

        return ll

    def _log_likelihood_single_process(self, k):
        """
        Helper function to compute the log likelihood of a single process
        :param k: process to compute likelihood for
        :return:
        """
        ll = 0

        # Get the likelihood of the datasets
        for ind,(S,_,_,_)  in enumerate(self.data_list):
            Rk = self.compute_rate(index=ind, proc=k)
            ll += self._poisson_log_likelihood(S[:,k], Rk)

        return ll


    def log_probability(self):
        """
        Compute the joint log probability of the data and the parameters
        :return:
        """
        lp = self.log_likelihood()

        # Get the parameter priors
        lp += self.bias_model.log_probability()
        lp += self.weight_model.log_probability()
        lp += self.impulse_model.log_probability()

        return lp

class DiscreteTimeNetworkHawkesModelGibbs(_DiscreteTimeNetworkHawkesModelBase, ModelGibbsSampling):
    _weight_class = SpikeAndSlabGammaWeights
    _parent_class = Parents

    def copy_sample(self):
        """
        Return a copy of the parameters of the model
        :return: The parameters of the model (A,W,\lambda_0, \beta)
        """
        return copy.deepcopy(self.get_parameters())

    def resample_model(self):
        """
        Perform one iteration of the Gibbs sampling algorithm.
        :return:
        """
        # Update the bias model given the parents assigned to the background
        self.bias_model.resample(
            data=np.concatenate([p.Z0 for (_,_,_,p) in self.data_list]))

        # Update the impulse model given the parents assignments
        self.impulse_model.resample(
            data=np.concatenate([p.Z for (_,_,_,p) in self.data_list]))

        # Update the weight model given the parents assignments
        self.weight_model.resample(
            model=self,
            N=np.atleast_1d(np.sum([N for (_,N,_,_) in self.data_list], axis=0)),
            Z=np.concatenate([p.Z for (_,_,_,p) in self.data_list]))

        # Update the parents.
        # THIS MUST BE DONE IMMEDIATELY FOLLOWING WEIGHT UPDATES!
        for _,_,_,p in self.data_list:
            p.resample(self.bias_model, self.weight_model, self.impulse_model)


class DiscreteTimeNetworkHawkesModelMeanField(_DiscreteTimeNetworkHawkesModelBase, ModelMeanField):
    _weight_class = GammaMixtureWeights
    _parent_class = Parents

    def meanfield_coordinate_descent_step(self):
        # Update the bias model given the parents assigned to the background
        self.bias_model.meanfieldupdate(
            EZ0=np.concatenate([p.EZ0 for (_,_,_,p) in self.data_list]))

        # Update the impulse model given the parents assignments
        self.impulse_model.meanfieldupdate(
            EZ=np.concatenate([p.EZ for (_,_,_,p) in self.data_list]))

        # Update the weight model given the parents assignments
        self.weight_model.meanfieldupdate(
            model=self,
            N=np.atleast_1d(np.sum([N for (_,N,_,_) in self.data_list], axis=0)),
            Z=np.concatenate([p.Z for (_,_,_,p) in self.data_list]))

        # Update the parents.
        # THIS MUST BE DONE IMMEDIATELY FOLLOWING WEIGHT UPDATES!
        for _,_,_,p in self.data_list:
            p.meanfieldupdate(self.bias_model, self.weight_model, self.impulse_model)