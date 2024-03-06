import autograd.numpy as np
import pylab as plt
import seaborn as snb

from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.optimize import minimize
from autograd import value_and_grad
from autograd import grad
from autograd.scipy.stats import norm
from autograd import hessian
from autograd.misc.optimizers import adam
from autograd.misc.flatten import flatten

def log_npdf(x, m, v):
    return -0.5*(x-m)**2/v - 0.5*np.log(2*np.pi*v)




#######################################################################################
# Neural network model with MAP inference
# Adapted from: # https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
########################################################################################

class NeuralNetworkMAP(object):

    def __init__(self, X, y, layer_sizes, likelihood, alpha=1., step_size=0.01, max_itt=1000, seed=0):

        # data
        self.X = X
        self.y = y

        # model and optimization parameters
        self.likelihood = likelihood(y)
        self.layer_sizes = layer_sizes
        self.step_size = step_size
        self.max_itt = max_itt
        self.alpha = alpha

        # random number genration
        self.seed=seed
        self.rng = np.random.default_rng(seed)
        
        # initialize parameters and optimize
        self.params = self.init_random_params()
        self.optimize()


    def init_random_params(self):
        """Build a list of (weights, biases) tuples,
        one for each layer in the net."""
        return [(np.sqrt(2/n) * self.rng.standard_normal((m, n)),   # weight matrix
                np.sqrt(2/n) * self.rng.standard_normal(n))      # bias vector
                for m, n in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    def neural_net_predict(self, params, inputs):
        """Implements a deep neural network for classification.
        params is a list of (weights, bias) tuples.
        inputs is an (N x D) matrix.
        returns logits."""
        for W, b in params:
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs# - logsumexp(outputs, axis=1, keepdims=True)

    def predict(self, inputs):
        return self.neural_net_predict(self.params, inputs)

    def log_prior(self, params):
        # implement a Gaussian prior on the weights
        flattened_params, _ = flatten(params)
        return  np.sum(log_npdf(flattened_params, 0., 1/self.alpha))
    
    def log_likelihood(self, params):     
        y = self.neural_net_predict(params, self.X)
        return self.likelihood.log_lik(y.ravel())

    def log_posterior(self, params):
        return self.log_prior(params) + self.log_likelihood(params)

    def optimize(self):
    
        # Define training objective and gradient of objective using autograd.
        def objective(params, iter):
            return -self.log_posterior(params)
            
        objective_grad = grad(objective)

        # optimize
        self.params = adam(objective_grad, self.params, step_size=self.step_size, num_iters=self.max_itt)

        return self

#######################################################################################
# Helper function for sampling multivariate Gaussians
########################################################################################


def generate_samples(mean, K, M, jitter=1e-8):
    """ returns M samples from a zero-mean Gaussian process with kernel matrix K
    
    arguments:
    K      -- NxN kernel matrix
    M      -- number of samples (scalar)
    jitter -- scalar
    returns NxM matrix
    """
    
    L = np.linalg.cholesky(K + jitter*np.identity(len(K)))
    zs = np.random.normal(0, 1, size=(len(K), M))
    fs = mean + np.dot(L, zs)
    return fs


#######################################################################################
# Kernels
########################################################################################

# in the code below tau represents the distance between to input points, i.e. tau = ||x_n - x_m||.
def squared_exponential(tau, kappa, lengthscale):
    return kappa**2*np.exp(-0.5*tau**2/lengthscale**2)

def matern12(tau, kappa, lengthscale):
    return kappa**2*np.exp(-tau/lengthscale)

def matern32(tau, kappa, lengthscale):
    return kappa**2*(1 + np.sqrt(3)*tau/lengthscale)*np.exp(-np.sqrt(3)*tau/lengthscale)

class StationaryIsotropicKernel(object):

    def __init__(self, kernel_fun, kappa=1., lengthscale=1.0):
        """
            the argument kernel_fun must be a function of three arguments kernel_fun(||tau||, kappa, lengthscale), e.g. 
            squared_exponential = lambda tau, kappa, lengthscale: kappa**2*np.exp(-0.5*tau**2/lengthscale**2)
        """
        self.kernel_fun = kernel_fun
        self.kappa = kappa
        self.lengthscale = lengthscale

    def contruct_kernel(self, X1, X2, kappa=None, lengthscale=None, jitter=1e-8):
        """ compute and returns the NxM kernel matrix between the two sets of input X1 (shape NxD) and X2 (MxD) using the stationary and isotropic covariance function specified by self.kernel_fun
    
        arguments:
            X1              -- NxD matrix
            X2              -- MxD matrix
            kappa           -- magnitude (positive scalar)
            lengthscale     -- characteristic lengthscale (positive scalar)
            jitter          -- non-negative scalar
        
        returns
            K               -- NxM matrix    
        """

        # extract dimensions 
        N, M = X1.shape[0], X2.shape[0]

        # prep hyperparameters
        kappa = self.kappa if kappa is None else kappa
        lengthscale = self.lengthscale if lengthscale is None else lengthscale

        # compute all the pairwise distances efficiently
        dists = np.sqrt(np.sum((np.expand_dims(X1, 1) - np.expand_dims(X2, 0))**2, axis=-1))
        
        # squared exponential covariance function
        K = self.kernel_fun(dists, kappa, lengthscale)
        
        # add jitter to diagonal for numerical stability
        if len(X1) == len(X2) and np.allclose(X1, X2):
            K = K + jitter*np.identity(len(X1))
                
        assert K.shape == (N, M), f"The shape of K appears wrong. Expected shape ({N}, {M}), but the actual shape was {K.shape}. Please check your code. "
        return K




#######################################################################################
# For plotting
########################################################################################

def plot_with_uncertainty(ax, Xp, mu, Sigma, sigma=0, color='g', color_samples='g', title="", num_samples=0):
    
    mean, std = mu.ravel(), np.sqrt(np.diag(Sigma) + sigma**2)

    
    # plot distribution
    ax.plot(Xp, mean, color=color, label='GP')
    ax.plot(Xp, mean + 2*std, color=color, linestyle='--')
    ax.plot(Xp, mean - 2*std, color=color, linestyle='--')
    ax.fill_between(Xp.ravel(), mean - 2*std, mean + 2*std, color=color, alpha=0.25, label='95% interval')
    
    # generate samples
    if num_samples > 0:
        fs = generate_samples(mu[:, None], Sigma, num_samples, 1e-6)
        ax.plot(Xp, fs, color=color_samples, alpha=.25)
    
    ax.set_title(title)

    if num_samples > 0:
        return fs
    

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
        
def eval_density_grid(density_fun, P=100, a=-5, b=5):
    x_grid = np.linspace(a, b, P)
    X1, X2 = np.meshgrid(x_grid, x_grid)
    XX = np.column_stack((X1.ravel(), X2.ravel()))
    return x_grid, density_fun(XX).reshape((P, P))


#######################################################################################
# compute classification error and std. error of the mean
########################################################################################


def compute_err(t, tpred):
    return np.mean(tpred.ravel() != t), np.std(tpred.ravel() != t)/np.sqrt(len(t))

#######################################################################################
# load subset of mnist data
########################################################################################

def load_MNIST_subset(filename, digits=[4,7], plot=True, subset=300, seed=0):

    data = np.load(filename)
    images = data['images']
    labels = data['labels']

    # we will only focus on binary classification using two digits
    idx = np.logical_or(labels == digits[0], labels == digits[1])
    
    # extract digits of interest
    X = images[idx, :]
    t = labels[idx].astype('float')
    
    # set labels to 0/1/2/...
    for i in range(len(digits)):
        t[t == digits[i]] = i
        
    # split into training/test
    N = len(X)
    Ntrain = int(0.5*N)
    Ntest = N - Ntrain
    np.random.seed(seed)
    train_idx = np.random.choice(range(N), size=Ntrain, replace=False)
    test_idx = np.setdiff1d(range(N), train_idx)

    Xtrain = X[train_idx, :]
    Xtest = X[test_idx, :]
    ttrain = t[train_idx]
    ttest = t[test_idx]

    # standardize training set
    Xm = Xtrain.mean(0)
    Xs = Xtrain.std(0)
    Xs[Xs == 0] = 1 # avoid division by zero for "always black" pixels

    Xtrain_std = (Xtrain - Xm)/Xs
    Xtest_std = (Xtest - Xm)/Xs


    # reduce dimensionality to 2D using principal component analysis (PCA)
    U, s, V = np.linalg.svd(Xtrain_std)

    # get eigenvectors corresponding to the two largest eigenvalues
    eigen_vecs = V[:2, :]
    eigen_vals = s[:2]

    # set-up projection matrix
    Pmat = eigen_vecs.T*(np.sqrt(Ntrain)/eigen_vals)

    # project and standize
    Phi_train =Xtrain_std@Pmat
    Phi_test = Xtest_std@Pmat

    # Let's only use a small subset of the training data
    Phi_train = Phi_train[:subset]
    ttrain = ttrain[:subset]

    return Phi_train, Phi_test, ttrain, ttest,
