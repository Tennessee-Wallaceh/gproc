"""
A number of python kernels, as defined in https://www.cs.toronto.edu/~duvenaud/cookbook/
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gamma, norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from functools import reduce

JITTER = 1e-5 # Add so-called jitter for stability of inversion

class InvertError(Exception):
    pass

<<<<<<< HEAD
def chol_inverse(symmetric_x, JITTER=1e-5):
=======
def chol_inverse(symmetric_x, jitter=JITTER):
>>>>>>> dbefb59d13b7b5b008bd5527a3ecb11ec65d4f38
    """
    Computes the Cholesky decomposition x=LL^T, and uses this to compute the inverse of X.
    Only valid for symmetric x.

    Parameters
    ----------
    symmetric_x: num_observations x num_observations numpy array
        no symmetry checks are performed 

    Returns
    ----------
    x_inv: num_observations x num_observations numpy array
        :math:`x^{-1}`, the inverse of symmetric_x
        
    chol: num_observations x num_observations numpy array
        matrix with lower triangular cholesky factor inside
    """
    dim_1 = symmetric_x.shape[0]
    counter = 0
    while True:
        try:
            counter +=1
            chol = cho_factor(symmetric_x + jitter * np.eye(dim_1), lower=True, check_finite=True)
            return cho_solve(chol, np.eye(dim_1)), chol[0]
        except LinAlgError:
            if counter > 5:
                raise InvertError('Attempted to invert matrix more than 5 times')
            jitter = jitter * 1.1
            pass

class BaseKernel:
    def __init__(self):
        pass

    def make_gram(self, x_1, x_2):
        raise NotImplementedError()
        
    def invert_gram(self):
        raise NotImplementedError()
        
    @staticmethod
    def constrain_params(self, unconstrained_param_array):
        raise NotImplementedError()
    @staticmethod
    def prior_log_pdf(self, param_array):
        raise NotImplementedError()
        
def squared_exponential(x_1, x_2, lengthscale=0.5, variance=1.0):
    """
    Also known as RBF.

    Parameters
    ----------
    x_1: N x D numpy array
    
    x_2: M x D numpy array

    lengthscale: float
        Lower lengthscale gives "wigglier" functions
    
    variance: float
        The average distance from the mean

    Returns
    ----------
    K: N x M matrix,
        The corresponding kernel matrix :math:`K_{ij} = k(x_i, x_j)`
    """
    sq_diffs = cdist(x_1, x_2, metric ='sqeuclidean')
    return variance * np.exp(-0.5 * sq_diffs / lengthscale)

class SquaredExponential(BaseKernel):
    param_dim = 2
    def __init__(self, lengthscale=0.5, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance
        super().__init__()
    
    def make_gram(self, x_1, x_2):
        return squared_exponential(x_1, x_2, self.lengthscale, self.variance)
    
    def get_params(self):
        return np.array([self.lengthscale, self.variance])
    
    def update_params(self, params):
        self.lengthscale = params[0]
        self.variance = params[1]

    @staticmethod
    def constrain_params(unconstrained_params):
        # if unconstrained_params.shape[0] != self.param_dim:
        #     raise AssertionError('Parameter array not the same size as kernel parameter dimension')
        return np.exp(unconstrained_params)
    
    @staticmethod
    def prior_log_pdf(constrained_params, d):
        prior = gamma.logpdf(constrained_params[0], a = 1, scale = np.sqrt(d))
        prior += gamma.logpdf(constrained_params[1], a = 1.2, scale = 1/0.2)
        return prior
    
def rational_quadratic(x_1, x_2, lengthscale=0.5, variance=1.0, weighting=1.0):
    """
    Rational Quadratic Kernel, equivalent to adding together many Squared Exponential kernels with different 
    lengthscales. Weight parameter determine relative weighting of large and small scale variations. When
    the weighting goes to infinity, RQ = SE.

    Parameters
    ----------
    x_1: N x D numpy array
    
    x_2: M x D numpy array

    lengthscale: float
        Lower lengthscale gives "wigglier" functions
    
    variance: float
        The average distance from the mean

    weighting: float
        Relative weighting of large and small scale variations

    Returns
    ----------
    K: N x M matrix,
        The corresponding kernel matrix :math:`K_{ij} = k(x_i, x_j)`
    """
    sq_diffs = cdist(x_1, x_2, metric = 'sqeuclidean')
    return variance * ( (1 + sq_diffs / 2*lengthscale * weighting) ** (-weighting) )

class RationalQuadratic(BaseKernel):
    param_dim = 3
    def __init__(self, lengthscale=0.5, variance=1.0, weighting=1.0):
        self.lengthscale = lengthscale
        self.variance = variance
        self.weighting = weighting
        super().__init__()
    
    def make_gram(self, x_1, x_2):
        return rational_quadratic(x_1, x_2, self.lengthscale, self.variance, self.weighting)
    
    def get_params(self):
        return np.array([self.lengthscale, self.variance, self.weighting])
    
    def update_params(self, params):
        self.lengthscale = params[0]
        self.variance = params[1]
        self.weighting = params[2]

    @staticmethod
    def constrain_params(unconstrained_params):
        # if unconstrained_params.shape[0] != self.param_dim:
        #     raise AssertionError('Parameter array not the same size as kernel parameter dimension')
        return np.exp(unconstrained_params)
    
    @staticmethod
    def prior_log_pdf(constrained_params, d):
        prior = gamma.logpdf(constrained_params[0], a = 1, scale = np.sqrt(d))
        prior += gamma.logpdf(constrained_params[1], a = 1.2, scale = 1/0.2)
        prior += gamma.logpdf(constrained_params[2], a = 0.00001, scale = 1/0.00001) # uninformative prior
        return prior
    

def periodic(x_1, x_2, lengthscale=0.5, variance=1.0, period=1.0):
    """
    The periodic kernel allows one to model functions which repeat themselves exactly.

    Parameters
    ----------
    x_1: N x D numpy array
    
    x_2: M x D numpy array

    lengthscale: float
        Lower lengthscale gives "wigglier" functions
    
    variance: float
        The average distance from the mean

    period: float
        Distance between repititions

    Returns
    ----------
    K: N x M matrix,
        The corresponding kernel matrix :math:`K_{ij} = k(x_i, x_j)`
    """
    diffs = cdist(x_1, x_2, metric = 'euclidean')
    return variance * np.exp(-2 * np.sin(np.pi * diffs / period)**2 / lengthscale)
                      
class Periodic(BaseKernel):
    param_dim = 3
    def __init__(self, lengthscale=0.5, variance=1.0, period=1.0):
        self.lengthscale = lengthscale
        self.variance = variance
        self.period = period
        super().__init__()
    
    def make_gram(self, x_1, x_2):
        return periodic(x_1, x_2, self.lengthscale, self.variance, self.period)
    
    def get_params(self):
        return np.array([self.lengthscale, self.variance, self.period])

    def update_params(self, params):
        self.lengthscale = params[0]
        self.variance = params[1]
        self.period = params[2]
    
    @staticmethod
    def constrain_params(unconstrained_params):
        # if unconstrained_params.shape[0] != self.param_dim:
        #     raise AssertionError('Parameter array not the same size as kernel parameter dimension')
        return np.exp(unconstrained_params)
        
    @staticmethod
    def prior_log_pdf(constrained_params, d):
        prior = gamma.logpdf(constrained_params[0], a = 1, scale = np.sqrt(d))
        prior += gamma.logpdf(constrained_params[1], a = 1.2, scale = 1/0.2)
        prior += gamma.logpdf(constrained_params[2], a = 0.00001, scale = 1/0.00001) # uninformative prior
        return prior
                             
                  
def locally_periodic(x_1, x_2, lengthscale_sqe=0.5, variance=1.0, lengthscale_p =0.5, period=1.0):
    """
    A squared exponential kernel multiplied by a periodic kernel. Allows one to model periodic functions
    which can vary slowly over time.

    Parameters
    ----------
    x_1: N x D numpy array
    
    x_2: M x D numpy array

    lengthscale: float
        Lower lengthscale gives "wigglier" functions
    
    variance: float
        The average distance from the mean

    period: float
        Distance between repititions

    Returns
    ----------
    K: N x M matrix,
        The corresponding kernel matrix :math:`K_{ij} = k(x_i, x_j)`
    """
    
    diffs = cdist(x_1, x_2, metric = 'euclidean')
    sq_diffs = cdist(x_1, x_2, metric = 'sqeuclidean')
    
    K_period = np.exp(-2 * np.sin(np.pi * diffs / period)**2 / lengthscale_p)
    K_sqe = np.exp(-0.5 * sq_diffs / lengthscale_sqe)
    return variance * np.multiply(K_period, K_sqe)

class LocallyPeriodic(BaseKernel):
    param_dim = 4
    def __init__(self, lengthscale_sqe=0.5, variance=1.0, lengthscale_p =0.5, period=1.0):
        self.lengthscale_sqe = lengthscale_sqe
        self.variance = variance
        self.lengthscale_p = lengthscale_p
        self.period = period
        super().__init__()
    
    def make_gram(self, x_1, x_2):
        gram = locally_periodic(x_1, x_2, self.lengthscale_sqe, self.variance, self.lengthscale_p, self.period)
        return gram 

    def get_params(self):
        return np.array([self.lengthscale_sqe, self.variance, self.lengthscale_p, self.period])
    
    def update_params(self, params):
        self.lengthscale_sqe = params[0]
        self.variance = params[1]
        self.lengthscale_p = arams[2]
        self.period = params[3]
    
    @staticmethod
    def constrain_params(unconstrained_params):
        # if unconstrained_params.shape[0] != self.param_dim:
        #     raise AssertionError('Parameter array not the same size as kernel parameter dimension')
        return np.exp(unconstrained_params)
        
    @staticmethod
    def prior_log_pdf(constrained_params, d):
        prior = gamma.logpdf(constrained_params[0], a = 1, scale = np.sqrt(d))
        prior += gamma.logpdf(constrained_params[1], a = 1.2, scale = 1/0.2)
        prior += gamma.logpdf(constrained_params[2], a = 0.00001, scale = 1/0.00001) # uninformative prior
        prior += gamma.logpdf(constrained_params[3], a = 0.00001, scale = 1/0.00001) # uninformative prior
        return prior
                             
def linear(x_1, x_2, constant_variance=0.5, variance=1.0, offset=1.0):
    """
    A linear kernel is a non-stationary kernel, which when used with a GP, is equivalent to
    Bayesian linear regression.
    
    Parameters
    ----------
    x_1: N x D numpy array
    
    x_2: M x D numpy array
    
    constant_variance: float
        Controls height of function at 0

    variance: float
        The average distance from the mean

    offset: float
        x coordinate of location all functions go through

    Returns
    ----------
    K: N x M matrix,
        The corresponding kernel matrix :math:`K_{ij} = k(x_i, x_j)`
    """
    return constant_variance + variance * np.dot(x_1 - offset, x_2.T - offset)

class Linear(BaseKernel):
    param_dim = 3
    def __init__(self, constant_variance=0.5, variance=1.0, offset=1.0):
        self.constant_variance = constant_variance
        self.variance = variance
        self.offset = offset
        super().__init__()
    
    def make_gram(self, x_1, x_2):
        gram = linear(x_1, x_2, self.constant_variance, self.variance, self.offset)
        return gram 
    
    def get_params(self):
        return np.array([self.constant_variance, self.variance, self.offset])
    
    def update_params(self, params):
        self.constant_variance = params[0]
        self.variance = params[1]
        self.offset = params[2]
    
    @staticmethod
    def constrain_params(unconstrained_params):
        # if unconstrained_params.shape[0] != self.param_dim:
        #     raise AssertionError('Parameter array not the same size as kernel parameter dimension')
        return np.concatenate((np.exp(unconstrained_params[0:2]), unconstrained_params[2].reshape(-1)))
        
    @staticmethod
    def prior_log_pdf(constrained_params, d):
        prior = gamma.logpdf(constrained_params[0], a = 1, scale = np.sqrt(d))
        prior += gamma.logpdf(constrained_params[1], a = 1.2, scale = 1/0.2)
        prior += norm.logpdf(constrained_params[2])
        return prior
    

                             
# Use below additive and multiplicative kernels to create arbirary numbers of kernels from above building blocks
def add_kernels(Kernels):
    # Function adds together a number of Kernel classes and returns 
    # an appropriate class
    class Additive(BaseKernel):
        param_dim = sum(k.param_dim for k in Kernels)
        def __init__(self, kernel_params):
            self.kernel_classes = Kernels
            self.kernels = [
                Kernel(**kernel_params[i])
                for i, Kernel in enumerate(self.kernel_classes)
            ]
            super().__init__()
        
        def make_gram(self, x_1, x_2):
            grams = [k.make_gram(x_1, x_2) for k in self.kernels]
            self.gram = sum(grams)
            return self.gram
        
        def get_params(self):
            params = np.zeros(0)
            for k in self.kernels:
                params = np.concatenate(params, k.get_params())
                
        def update_params(self, params):
            dim_count = 0
            for k in self.kernels:
                k.update_params(params[dim_count:(dim_count + k.param_dim)])
                dim_count += k.param_dim
        
        @staticmethod
        def constrain_params(self, unconstrained_params):
            #if unconstrained_params.shape[0] != self.param_dim:
                #raise AssertionError('Parameter array not the same size as kernel parameter dimension')
                
            constrained_params = np.zeros(0)
            dim_count = 0
            for k in self.kernels:
                constrained_params = np.concatenate(
                    (constrained_params,
                    k.constrain_params(unconstrained_params[dim_count:(dim_count + k.param_dim)]))
                    )
                dim_count += k.param_dim
            return constrained_params
        
        @staticmethod
        def prior_log_pdf(self, constrained_params, d):
            prior = 0
            dim_count = 0
            for k in self.kernels:
                prior += k.prior_log_pdf(constrained_params[dim_count:(dim_count + k.param_dim)], d)
                dim_count += k.param_dim
            return prior

    return Additive

def multiply_kernels(Kernels):
    class Multiplicative(BaseKernel):
        param_dim = sum(k.param_dim for k in Kernels)
        def __init__(self, kernel_params):
            self.kernel_classes = Kernels
            self.kernels = [
                Kernel(**kernel_params[i])
                for i, Kernel in enumerate(self.kernel_classes)
            ]
            super().__init__()
        
        def make_gram(self, x_1, x_2):
            grams = [k.make_gram(x_1, x_2) for k in self.kernels]
            self.gram = reduce(np.multiply, grams)
            return self.gram
        
        def get_params(self):
            params = np.zeros(0)
            for k in self.kernels:
                params = np.concatenate(params, k.get_params())
                
        def update_params(self, params):
            dim_count = 0
            for k in self.kernels:
                k.update_params(params[dim_count:(dim_count + k.param_dim)])
                dim_count += k.param_dim
        
        @staticmethod
        def constrain_params(self, unconstrained_params):
            #if unconstrained_params.shape[0] != self.param_dim:
                #raise AssertionError('Parameter array not the same size as kernel parameter dimension')
                
            constrained_params = np.zeros(0)
            dim_count = 0
            for k in self.kernels:
                constrained_params = np.concatenate((constrained_params, k.constrain_params(unconstrained_params[dim_count:(dim_count + k.param_dim)])))
                dim_count += k.param_dim
            return constrained_params
        
        @staticmethod
        def prior_log_pdf(self, constrained_params, d):
            prior = 0
            dim_count = 0
            for k in self.kernels:
                prior += k.prior_log_pdf(constrained_params[dim_count:(dim_count + k.param_dim)], d)
                dim_count += k.param_dim
            return prior

    return Multiplicative