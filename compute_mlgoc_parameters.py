import numpy as np
import math
import scipy.integrate as integrate


def compute_mlgoc_parameters(alpha_tilde, lambda_tilde, r_star, r_hat_star):
    """ Computes the active contour, the phase field and the MRF parameters of the 'gas of circles' model
    """
    alpha_c = alpha_tilde
    lambda_c = lambda_tilde*r_star
    beta_c = compute_beta_tilde(alpha_tilde, lambda_tilde, r_star, r_hat_star, 'marie')
    parameters_c = {'alpha':alpha_c, 'beta': beta_c, 'lambda': lambda_c, 'r_star': r_star, 'd':r_star/r_hat_star, \
                    'epsilon':r_star/r_hat_star, 'D': None, 'discrete': 1, 'marie': 1}

    alpha_pf = 0.75 * alpha_c
    beta_pf = 0.25 * beta_c
    lambda_pf = lambda_c # w=4
    D_pf = lambda_c # w=4
    parameters_pf = {'alpha': alpha_pf, 'beta': beta_pf, 'lambda': lambda_pf, 'r_star': r_star, 'd': r_star / r_hat_star, \
                    'epsilon': r_star / r_hat_star, 'D': D_pf, 'discrete': 1, 'marie': 1}

    alpha_m = 0.5 * alpha_c
    beta_m = 0.25 * beta_c
    D_m = (math.pi / 16) * lambda_c
    lambda_m = - lambda_pf / 4

    parameters_m = {'alpha': alpha_m, 'beta': beta_m, 'lambda': lambda_m, 'r_star': r_star, 'd': r_star / r_hat_star, \
                    'epsilon': r_star / r_hat_star, 'D': D_m, 'discrete': 1, 'marie': 1}
    return [parameters_c, parameters_pf, parameters_m]


def compute_beta_tilde(alpha_tildes, lambda_tilde, r_star, r_hat_star, interaction_type):
    # _, r0s = np.meshgrid(alpha_tildes,r_star)
    fs = eval_F10(r_star, r_star/r_hat_star, r_star/r_hat_star, interaction_type)
    # _, Fs = np.meshgrid(alpha_tildes, fs)

    return (lambda_tilde + alpha_tildes) * r_star / fs


def eval_F10(r0s, d_mins, epsilons, interaction_type):
    # vF10 = np.zeros(np.shape(d_mins))
    # for i in range(len(d_mins)):
    #     d_min = d_mins[i]
    #     epsilon = epsilons[i]
    #     r0 = r0s[i]
    #     vF10[i] = 2 * integrate.quad(integrand_F10, 0, math.pi, (r0, d_min, epsilon, interaction_type) )
    vF10, _ = integrate.quad(integrand_F10, 0, math.pi, (r0s, d_mins, epsilons, interaction_type) )
    return 2.0*vF10


def integrand_F10(x, r0, d_min, epsilon, interaction_type):
    s_a_2 = abs(np.sin(x/2))
    x0 = 2 *r0 *s_a_2
    p = phi(x0, d_min, epsilon, interaction_type)
    p_p = phi_p(x0, d_min, epsilon, interaction_type)
    return r0 * np.cos(x) * (p+r0*s_a_2*p_p)


def phi(x, d_min, epsilon, interaction_type):
    f = np.zeros(np.shape(x))
    if interaction_type.lower() == 'exponential':
        f = np.exp(-x/d_min)
    elif interaction_type.lower() == 'marie':
        f = (x>d_min+epsilon)*0.0 + (x<d_min-epsilon)*1.0 + (x<d_min+epsilon)*(x>d_min-epsilon)*0.5*(1 - (x-d_min)/epsilon - np.sin(math.pi*(x-d_min)/epsilon)/math.pi)

    return f


def phi_p(x, d_min, epsilon, interaction_type):
    f = np.zeros(np.shape(x))
    if interaction_type.lower() == 'exponential':
        f = -1.0/d_min * np.exp(-x / d_min)
    elif interaction_type.lower() == 'marie':
        f = (x > d_min + epsilon) * 0.0 + (x < d_min - epsilon) * 0.0 + (x < d_min + epsilon) * (
        x > d_min - epsilon) * 0.5 * (-1-np.cos(math.pi*(x-d_min)/epsilon))/epsilon

    return f