"""
Author:

    Oliver Sheridan-Methven, October 2020.

Description:

    The various plots for the article.
"""

import plotting_configuration
import matplotlib.pylab as plt
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad as integrate
from approximate_random_variables.approximate_gaussian_distribution import construct_piecewise_constant_approximation, construct_symmetric_piecewise_polynomial_approximation
from mpmath import mp

mp.dps = 50

norm_inv = norm.ppf


def plot_piecewise_constant_approximation(savefig=False):
    u = np.linspace(0, 1, 10000)[1:-1]
    norm_inv_approx = construct_piecewise_constant_approximation(norm.ppf, 8)
    plt.clf()
    plt.plot(u, norm_inv(u), 'k--', label=r'$\Phi^{-1}(x)$')
    plt.plot(u, norm_inv_approx(u), 'k,', label=r'__nolegend__')
    plt.plot([], [], 'k-', label=r'$Q(x)$')
    plt.xlabel(r"$x$")
    plt.xticks([0, 1])
    plt.yticks([-3, 0, 3])
    plt.ylim(-3, 3)
    plt.legend(frameon=False)
    if savefig:
        plt.savefig('piecewise_constant_gaussian_approximation.pdf', format='pdf', bbox_inches='tight', transparent=True)


def plot_piecewise_constant_error(savefig=False):
    res = {1 << i: {} for i in range(1, 4)}
    for n in [1 << i for i in range(11)]:
        norm_inv_approx = construct_piecewise_constant_approximation(norm.ppf, n)
        discontinuities = np.linspace(0, 1, n + 1)
        for p in res:
            p_norm = integrate(lambda u: (norm_inv(u) - norm_inv_approx(u)) ** p, 0, 1, points=discontinuities, limit=50 + 10 * n)[0] ** (1.0 / p)
            res[p][n] = p_norm

    plt.clf()
    markers = (i for i in {'o', 's', 'v'})
    for p in res:
        n, p_norm = zip(*res[p].items())
        marker = next(markers)
        plt.plot(n, p_norm, 'k{}'.format(marker), label=r'$p = {}$'.format(p))
        plt.plot(n, p_norm, 'k:', label=r'__nolegend__')
        q = np.linspace(2, np.log2(n[-1]))  # For the analytic bound from Giles
        x = 2.0 ** q
        y = 2.0 ** (-q / p) * q ** -0.5
        y = y / y[-1] * p_norm[-1]  # Rescaled
        plt.plot(x, y, 'k--', label='__nolegend__')
    plt.plot([], [], 'k--', label=r'$O(2^{-q/p} q^{-1/2})$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$\norm{Z - \tilde{Z}}_p$')
    plt.xlabel('Intervals')
    plt.legend(frameon=False, handlelength=1, borderaxespad=0)
    if savefig:
        plt.savefig('piecewise_constant_gaussian_approximation_error.pdf', format='pdf', bbox_inches='tight', transparent=True)


def plot_piecewise_linear_gaussian_approximation(savefig=False):
    u = np.linspace(0, 1, 10000)[1:-1]
    norm_inv_approx = construct_symmetric_piecewise_polynomial_approximation(norm.ppf, n_intervals=5, polynomial_order=1)
    plt.clf()
    plt.plot(u, norm_inv(u), 'k--', label=r'$\Phi^{-1}(x)$')
    plt.plot(u, norm_inv_approx(u), 'k,', label=r'__nolegend__')
    plt.plot([], [], 'k-', label=r'$Q(x)$')
    plt.xlabel(r"$x$")
    plt.xticks([0, 1])
    plt.yticks([-3, 0, 3])
    plt.ylim(-3, 3)
    plt.legend(frameon=False)
    if savefig:
        plt.savefig('piecewise_linear_gaussian_approximation.pdf', format='pdf', bbox_inches='tight', transparent=True)


def plot_piecewise_linear_gaussian_approximation_error_singular_interval(savefig=False):
    cdf = mp.ncdf
    pdf = mp.npdf
    sqrt = mp.sqrt
    pi = mp.pi
    fabs = mp.fabs
    inf = mp.inf
    log10 = mp.log10
    log = mp.log
    delta = np.concatenate([np.logspace(-6, -1, 25), np.logspace(-1, np.log10(0.39894), 25)])
    z = norm.ppf(delta)
    p = 2
    integral = []
    for z_d in z:
        # Getting a numeric estimate.
        a = 0
        b = d = cdf(z_d)
        # Evaluating the analytic expression.
        b_analytic = 6.0 / (b - a) ** 3 * (cdf(sqrt(2) * z_d) / sqrt(pi) - (a + b) * pdf(z_d))
        a_analytic = 2.0 * pdf(z_d) / d - 3.0 * cdf(sqrt(2) * z_d) / (sqrt(pi) * d ** 2)
        # Evaluating the error via the exact integral. (The integrals need to be moderately well scaled).
        v_1 = fabs(z_d - a_analytic - b_analytic * cdf(z_d)) ** p * pdf(z_d)
        v_2 = v_1 * mp.quad(lambda z: v_1 ** -1 * fabs(z - a_analytic - b_analytic * cdf(z)) ** p * pdf(z), [-inf, z_d])
        e_integral = v_2
        integral.append(e_integral)
    x = delta
    y_integral = integral
    y1 = x
    y2 = x * np.log(1.0 / (np.sqrt(2.0 * np.pi) * x)) ** (-p / 2.0)
    plt.clf()
    plt.plot(2*x, y_integral, 'k-', label='__nolegend__')
    plt.plot(2*x, y2 / [y2[0] / y_integral[0]], 'k-', dashes=(10, 10), label=r'$O(r^{K-1} {\log}^{-p/2}(r^{1-K}\sqrt{2/\pi}))$')
    plt.plot(2*x, y1 / [y1[0] / y_integral[0]], 'k-', dashes=(3,3), label=r'$o(r^{K-1})$')
    plt.xlabel(r'$r^{K-1}$')
    plt.ylabel(r'$\int_{I_K} \lvert \Phi^{-1}(u) - D(u)\rvert^p \dd{u}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(frameon=False, loc='upper left', borderaxespad=0, handletextpad=0.5)
    plt.xlim(1e-6, 1e-1)
    plt.ylim(1e-8, 1e-1)
    if savefig:
        plt.savefig('piecewise_linear_gaussian_approximation_error_singular_interval.pdf', format='pdf', bbox_inches='tight', transparent=True)

def plot_piecewise_linear_gaussian_approximation_error_singular_interval(savefig=False):
    polynomial_orders = range(6)
    interval_sizes = [2, 4, 8, 16]
    plt.clf()
    for n_intervals in interval_sizes:
        rmse = [None] * len(polynomial_orders)
        for polynomial_order in polynomial_orders:
            approximate_inverse_gaussian_cdf = construct_symmetric_piecewise_polynomial_approximation(norm.ppf, n_intervals + 1, polynomial_order)  # +1 as we have the 0 interval which is measure 0.
            discontinuities = [0.5 ** (i + 2) for i in range(n_intervals)]  # Makes the numerical integration involved in the RMSE easier.
            rmse[polynomial_order] = integrate(lambda u: 2.0 * (norm.ppf(u) - approximate_inverse_gaussian_cdf(u)) ** 2, 0, 0.5, points=discontinuities)[0] ** 0.5
        plt.plot(polynomial_orders, rmse, 'ko:', label='__nolengend__')
        plt.plot([], [], 'ko', label=n_intervals)
        plt.gca().text(polynomial_orders[-1] + 0.2, rmse[-1], str(n_intervals), va='center')
    plt.yscale('log')
    plt.ylabel(r'$\lVert Z - \tilde{Z}\rVert_2$')
    plt.xlabel('Polynomial order')
    plt.xlim(None, 5.7)
    plt.ylim(1e-4, 1e0)
    plt.xticks(polynomial_orders)
    if savefig:
        plt.savefig('piecewise_linear_gaussian_approximation_error.pdf', format='pdf', bbox_inches='tight', transparent=True)
