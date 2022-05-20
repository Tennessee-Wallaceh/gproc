"""Provides functions for performing Expectation Propagation approximation to GP classification posteriors."""

from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve, cholesky, sqrtm, det
from scipy.linalg import norm as la_norm
import numpy as np

JITTER = 1e-5 

def expectation_propagation_probit(observed_y, gram, max_iterations=100, tol=1e-5):
    """
    Computes the expectation propagation (Algorithm 3.5) approximation to the latent
    function implied by the model:

        p(y_i | f_i) = norm_cdf(y_i * f_i)
        p(f | gram) = normal(0, gram)

    We target the posterior:

        p(f | y) = Z * p(y | f) * p(f | gram)

    With an approximation q(f) = normal(mu, Sig).

    :param y: num_observations x 1 numpy array containing 1 or -1
    :param gram: num_observations x num_observations numpy array
    """
    
    # Data size and gram matrix relabelling
    N = y.shape[0]
    K = gram
    
    # Initialise parameter arrays
    nu_site = np.zeros(N)
    tau_site = np.zeros(N)
    nu_cav = np.zeros(N)
    tau_cav = np.zeros(N)
    z = np.zeros(N)
    Sig = np.copy(gram)
    mu_proposed = np.zeros(N)
    
    def cavity_params(Sig, tau_site, nu_site, mu, i):
        """Compute approximate cavity parameters"""
        tau_cav_i = Sig[i, i]**(-1) - tau_site[i]
        nu_cav_i = Sig[i, i]**(-1) * mu[i] - nu_site[i]
        return tau_cav_i, nu_cav_i
    
    def marginal_moments(nu_cav, tau_cav, y, i):
        """Compute marginal moments"""
        z_i = ( y[i] * nu_cav[i] / tau_cav[i] ) / np.sqrt( 1 + tau_cav[i]**(-1) )
        mu_hat = ( nu_cav[i] / tau_cav[i] ) + ( y[i] * tau_cav[i]**(-1) * norm.pdf(z[i]) ) / ( norm.cdf(z[i]) * np.sqrt( 1 + tau_cav[i]**(-1) ) ) 
        var_hat = tau_cav[i]**(-1)  - ( ( tau_cav[i]**(-2) * norm.pdf(z[i]) ) / ( ( 1 + tau_cav[i]**(-1) ) * norm.cdf(z[i]) ) ) * ( z[i] + ( norm.pdf(z[i]) / norm.cdf(z[i]) ) ) 
        return z_i, mu_hat, var_hat
    
    def site_params(var_hat, tau_cav, tau_site, i):
        """Update site parameters"""
        tau_delta = var_hat**(-1) - tau_cav[i] - tau_site[i]
        tau_site_i = tau_site[i] + tau_delta
        nu_site_i = var_hat**(-1) * mu_hat - nu_cav[i]
        return tau_delta, tau_site_i, nu_site_i
    
    def posterior_params(Sig, tau_delta, nu_site):
        """Update approximate posterior parameters"""
        Sig = Sig - ( ( tau_delta**(-1) + Sig[i, i] )**(-1) * Sig[:, i].dot(Sig[:, i]) )
        mu = Sig.dot(nu_site)
        return Sig, mu
    
    def re_posterior_params(tau_site, K, JITTER=1e-5):
        """Recompute approximate posterior parameters"""
        S_site_sqrt = np.diag(np.sqrt(tau_site))
        L = cho_factor(JITTER * np.eye(N) + S_site_sqrt.dot(K).dot(S_site_sqrt), lower=True, check_finite=True)
        V = cho_solve(L, S_site_sqrt.dot(K))
        Sig = K - V.T.dot(V)
        mu_proposed = Sig.dot(nu_site)
        return L[0], Sig, mu_proposed
    
    def mark_lik_ev(tau_cav, L, tau_site, nu_site, nu_cav):
        """Compute marginal likelihood log evidence"""
        one = np.sum( np.log( np.diagonal(L) ) )
        two = 0.5 * nu_site.dot( Sig - np.diag( (tau_cav + tau_site)**(-1) ) ).dot(nu_site)
        three = np.sum( np.log( norm.cdf( y.dot( nu_cav / tau_cav ) / ( np.sqrt( 1 + tau_cav**(-1) ) ) ) ) )
        four = 0.5 * np.log( 1 + tau_site.dot( tau_cav**(-1) ) )
        five = 0.5 * (nu_cav / tau_cav).dot(T).dot( np.diag( (tau_cav + tau_site)**(-1) ) ).dot( S.dot(nu_cav / tau_cav) - 2 * nu_site)
        return one + two + three + four + five
    
    for i in range(1, max_iterations):
        mu = mu_proposed
        
        for i in range(N):
            # Compute approximate cavity parameters
            tau_cav[i], nu_cav[i] = cavity_params(Sig, tau_site, nu_site, mu, i)
            
            # Compute marginal moments
            z[i], mu_hat, var_hat = marginal_moments(nu_cav, tau_cav, y, i)

            # Update site parameters
            tau_delta, tau_site[i], nu_site[i] = site_params(var_hat, tau_cav, tau_site, i)

            # Update approximate posterior parameters
            Sig, mu = posterior_params(Sig, tau_delta, nu_site)

        # Recompute the approximate posterior parameters        
        L, Sig, mu_proposed = re_posterior_params(tau_site, K, JITTER=1e-5)

        # Check convergence condition
        if la_norm(mu_proposed, mu) < tol:
            converged = True
            break
    
    # Compute marginal likelihood log evidence
    log_evidence = mark_lik_ev(tau_cav, L, tau_site, nu_site, nu_cav, tau_cav)
    
    return nu_site, v_site, log_evidence, converged