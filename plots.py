"""
Author:

    Oliver Sheridan-Methven, October 2020.

Description:

    The various plots for the article.
"""

import plotting_configuration
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, ncx2
from scipy.integrate import quad as integrate
from approximate_random_variables.approximate_gaussian_distribution import construct_piecewise_constant_approximation, construct_symmetric_piecewise_polynomial_approximation, rademacher_approximation
from approximate_random_variables.approximate_non_central_chi_squared import construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation
from mpmath import mp, mpf
from timeit import default_timer as timer
from functools import wraps
from progressbar import progressbar


def time_function(func):
    """ A decorator to time a function. """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = timer()
        results = func(*args, **kwargs)
        elapsed_time = timer() - start_time
        return results, elapsed_time

    return wrapper


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
    plt.plot([], [], 'k-', label=r'$D(x)$')
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
    plt.plot(2 * x, y_integral, 'k-', label='__nolegend__')
    plt.plot(2 * x, y2 / [y2[0] / y_integral[0]], 'k-', dashes=(10, 10), label=r'$O(r^{K-1} {\log}^{-p/2}(r^{1-K}\sqrt{2/\pi}))$')
    plt.plot(2 * x, y1 / [y1[0] / y_integral[0]], 'k-', dashes=(3, 3), label=r'$o(r^{K-1})$')
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


def produce_geometric_brownian_motion_paths(dt, method=None, approx=None):
    """
    Perform path simulations of a geometric Brownian motion.
    :param dt: Float. (Fraction of time).
    :param method: Str.
    :param approx: List.
    :return: List. [x_fine_exact, x_coarse_exact, x_fine_approx, x_coarse_approx]
    """
    assert isinstance(dt, float) and np.isfinite(dt) and dt > 0 and (1.0 / dt).is_integer()
    assert isinstance(method, str) and method in ['euler_maruyama', 'milstein']
    assert approx is not None
    # The parameters.

    x_0 = 1.0
    mu = 0.05
    sigma = 0.2
    T = 1.0

    dt = dt * T
    t_fine = dt
    t_coarse = 2 * dt
    sqrt_t_fine = t_fine ** 0.5
    w_coarse_exact = 0.0
    w_coarse_approx = 0.0

    x_fine_exact = x_0
    x_coarse_exact = x_0
    x_fine_approx = x_0
    x_coarse_approx = x_0
    n_fine = int(1.0 / dt)

    update_coarse = False

    x_0, mu, sigma, T, dt, t_fine, t_coarse, sqrt_t_fine, w_coarse_exact, w_coarse_approx = [mpf(i) for i in [x_0, mu, sigma, T, dt, t_fine, t_coarse, sqrt_t_fine, w_coarse_exact, w_coarse_approx]]
    fabs = mp.fabs

    path_update = None
    if method == 'euler_maruyama':
        path_update = lambda x, w, t: x + mu * x * t + sigma * x * w
    elif method == 'milstein':
        path_update = lambda x, w, t: x + mu * x * t + sigma * x * w + 0.5 * sigma * sigma * (w * w - t)
    assert path_update is not None

    for n in range(n_fine):
        u = np.random.uniform()
        z_exact = norm.ppf(u)
        z_approx = approx(u)
        z_approx = z_approx if isinstance(z_approx, float) else z_approx[0]
        w_fine_exact = sqrt_t_fine * z_exact
        w_fine_approx = sqrt_t_fine * z_approx
        w_coarse_exact += w_fine_exact
        w_coarse_approx += w_fine_approx

        x_fine_exact = path_update(x_fine_exact, w_fine_exact, t_fine)
        x_fine_approx = path_update(x_fine_approx, w_fine_approx, t_fine)
        if update_coarse:
            x_coarse_exact = path_update(x_coarse_exact, w_coarse_exact, t_coarse)
            x_coarse_approx = path_update(x_coarse_approx, w_coarse_approx, t_coarse)
            w_coarse_exact *= 0.0
            w_coarse_approx *= 0.0
        update_coarse = not update_coarse  # We toggle to achieve pairwise summation.
    assert not update_coarse  # This should have been the last thing we did.

    return [x_fine_exact, x_coarse_exact, x_fine_approx, x_coarse_approx]


def plot_variance_reduction(savefig=False):
    deltas = [2.0 ** -i for i in range(1, 7)]
    inverse_norm = norm.ppf
    piecewise_constant = construct_piecewise_constant_approximation(inverse_norm, n_intervals=1024)
    piecewise_linear = construct_symmetric_piecewise_polynomial_approximation(inverse_norm, n_intervals=16, polynomial_order=1)
    piecewise_cubic = construct_symmetric_piecewise_polynomial_approximation(inverse_norm, n_intervals=16, polynomial_order=3)
    approximations = {'constant': piecewise_constant, 'linear': piecewise_linear, 'cubic': piecewise_cubic, 'rademacher': rademacher_approximation}
    markers = {'original': 'd', 'constant': 'o', 'linear': 'v', 'cubic': 's', 'rademacher': 'x'}

    results = {method: {term: {} for term in ['original'] + list(approximations.keys())} for method in ['euler_maruyama', 'milstein']}  # Store the values of delta and the associated data.
    time_per_level = 3.0
    paths_min = 64
    for method in results:
        for approx_name, approx in approximations.items():
            for dt in deltas:
                _, elapsed_time_per_path = time_function(produce_geometric_brownian_motion_paths)(dt, method, approx)
                paths_required = int(time_per_level / elapsed_time_per_path)
                if paths_required < paths_min:
                    print("More time required for {} and {} with dt={}".format(method, approx_name, dt))
                    break

                originals, corrections = [[None for i in range(paths_required)] for j in range(2)]
                for path in range(paths_required):
                    x_fine_exact, x_coarse_exact, x_fine_approx, x_coarse_approx = produce_geometric_brownian_motion_paths(dt, method, approx)
                    originals[path] = x_fine_exact - x_coarse_exact
                    corrections[path] = min((x_fine_exact - x_coarse_exact) - (x_fine_approx - x_coarse_approx), (x_fine_exact - x_fine_approx) - (x_coarse_exact - x_coarse_approx), sum([x_fine_exact, -x_coarse_exact, -x_fine_approx, x_coarse_approx]))  # might need revising for near machine precision.
                originals, corrections = [[j ** 2 for j in i] for i in [originals, corrections]]
                for name, values in [['original', originals], [approx_name, corrections]]:
                    mean = np.mean(values)
                    std = np.std(values) / (len(values) ** 0.5)
                    results[method][name][dt] = [mean, std]

    for method in results:
        plt.clf()
        for approx_name in results[method]:
            x, y = zip(*results[method][approx_name].items())
            y, y_std = list(zip(*y))
            y_error = 1 * np.array(y_std)
            plt.errorbar(x, y, y_error, None, 'k{}:'.format(markers[approx_name]))
        plt.xscale('log', basex=2)
        plt.yscale('log', basey=2)
        plt.xlabel(r'Fine time increment $\delta^{\mathrm{f}}$')
        plt.ylabel('Variance')
        y_min_base_2 = 50
        plt.ylim(2 ** -y_min_base_2, 2 ** -10)
        plt.yticks([2 ** -i for i in range(10, y_min_base_2 + 1, 10)])
        plt.xticks(deltas)
        if savefig:
            plt.savefig('variance_reduction_{}_scheme.pdf'.format(method), format='pdf', bbox_inches='tight', transparent=True)


def rmse_of_non_central_chi_squared_polynomial_approximations():
    lambdas = [1, 5, 10, 50, 100, 200]
    nus = [1, 5, 10, 50, 100]
    poly_orders = [1, 3, 5]
    n_intervals = 16
    results = {poly_order: {nu: {} for nu in nus} for poly_order in poly_orders}
    for poly_order in poly_orders:
        for nu in nus:
            ncx2_approx = construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation(dof=nu, n_intervals=n_intervals + 1, polynomial_order=poly_order)
            discontinuities = sorted([0.5 ** (i + 2) for i in range(n_intervals)] + [0.5] + [1.0 - 0.5 ** (i + 2) for i in range(n_intervals)])
            for l in lambdas:
                rmse = integrate(lambda u: (ncx2.ppf(u, df=nu, nc=l) - ncx2_approx(u, non_centrality=l)) ** 2, 0, 1, points=discontinuities, limit=50 + 10 * len(discontinuities))[0] ** 0.5
                results[poly_order][nu][l] = rmse

    for poly_order, result in results.items():
        df = pd.DataFrame(result)
        df.index = df.index.rename('lambda')
        df.columns = df.columns.rename('nu')
        print(poly_order, df.min().min(), df.max().max())
        print(round(df, 3))
        print('\n')
        print(round(df, 3).apply(lambda x: ' & '.join([str(i) for i in list(x)]) + r' \\', axis=1))
        print('\n' * 3)


def produce_cox_ingersoll_ross_paths(dt, approximations=None, **kwargs):
    assert isinstance(dt, float) and np.isfinite(dt) and dt > 0 and (1.0 / dt).is_integer()
    assert approximations is not None
    # The parameters.
    params = kwargs
    kappa, theta, sigma = params['kappa'], params['theta'], params['sigma']
    T = 1.0
    x_0 = 1.0
    dt = dt * T
    sqrt_t = dt ** 0.5
    c1 = 4.0 * kappa / (sigma ** 2 * (1.0 - np.exp(-kappa * dt)))
    c2 = c1 * np.exp(-kappa * dt)
    df = 4.0 * kappa * theta / (sigma ** 2)

    euler_maruyama_update = lambda x, w, t: x + kappa * (theta - x) * t + sigma * np.sqrt(np.fabs(x)) * w
    exact_update = lambda u, x: ncx2.ppf(u, df=df, nc=x * c2) / c1
    approximate_update = lambda u, x, approx: approx(u, non_centrality=x * c2)[0] / c1

    x_exact = x_0
    x_euler_maruyama = x_0
    x_approximations = [x_0] * len(approximations)

    n_increments = int(1.0 / dt)

    for n in range(n_increments):
        u = np.random.uniform()
        z = norm.ppf(u)
        dw = sqrt_t * z
        x_euler_maruyama = euler_maruyama_update(x_euler_maruyama, dw, dt)
        x_exact = exact_update(u, x_exact)
        x_approximations = [approximate_update(u, x_approximate, approx) for approx, x_approximate in zip(approximations, x_approximations)]

    return [x_euler_maruyama, x_exact, *x_approximations]


def plot_variance_reduction_cir_process(savefig=False):
    deltas = [0.5 ** i for i in range(8)]
    poly_orders = {'linear': 1, 'cubic': 3}
    poly_markers = (i for i in ['s', 'd'])
    results = {k: {} for k in ['exact', 'euler_maruyama'] + list(poly_orders.keys())}
    markers = {**{'exact': 'o', 'euler_maruyama': 'v'}, **{k: next(poly_markers) for k in poly_orders}}
    params = {'kappa': 0.5, 'theta': 1.0, 'sigma': 1.0}
    nu = 4.0 * params['kappa'] * params['theta'] / (params['sigma'] ** 2)
    approximations = [construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation(dof=nu, polynomial_order=poly_order) for poly_order in [1, 3]]
    time_per_level = 5.0
    paths_min = 64
    for dt in progressbar(deltas):
        _, elapsed_time_per_path = time_function(produce_cox_ingersoll_ross_paths)(dt, approximations, **params)
        paths_required = int(time_per_level / elapsed_time_per_path)
        if paths_required < paths_min:
            print("More time required for dt={}".format(dt))
            break
        exacts, euler_maruyamas, linears, cubics = [[None for i in range(paths_required)] for j in range(4)]
        for path in range(paths_required):
            x_euler_maruyama, x_exact, x_linear, x_cubic = produce_cox_ingersoll_ross_paths(dt, approximations, **params)
            exacts[path] = x_exact
            euler_maruyamas[path] = x_exact - x_euler_maruyama
            linears[path] = x_exact - x_linear
            cubics[path] = x_exact - x_cubic
        exacts, euler_maruyamas, linears, cubics = [[j ** 2 for j in i] for i in [exacts, euler_maruyamas, linears, cubics]]
        for name, values in {'exact': exacts, 'euler_maruyama': euler_maruyamas, 'linear': linears, 'cubic': cubics}.items():
            mean = np.mean(values)
            std = np.std(values) / (len(values) ** 0.5)
            results[name][dt] = [mean, std]

    plt.clf()
    for name in results:
        x, y = zip(*results[name].items())
        y, y_std = list(zip(*y))
        y_error = 1 * np.array(y_std)
        plt.errorbar(x, y, y_error, None, 'k{}:'.format(markers[name]))
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=2)
    plt.xticks(x)
    plt.ylim(2 ** -25, 2 ** 2)
    plt.yticks([2 ** -i for i in range(0, 30, 5)])
    plt.xlabel(r'Time increment $\delta$')
    plt.ylabel('Variance')
    if savefig:
        plt.savefig('variance_reduction_cir_process.pdf', format='pdf', bbox_inches='tight', transparent=True)


def plot_non_central_chi_squared_polynomial_approximation(save_figure=False):
    dof = 1.0
    ncx2_approx = construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation(dof, n_intervals=4 + 1)
    u = np.linspace(0.0, 1.0, 100000)[:-1]  # Excluding the end points.
    u_approx = np.linspace(0.0, 1.0, 1000)[:-1]
    non_centralities = [1.0, 10.0, 20.0]
    plt.clf()
    for non_centrality in non_centralities:
        plt.plot(u, ncx2.ppf(u, df=dof, nc=non_centrality), 'k--')
        plt.plot(u_approx, ncx2_approx(u_approx, non_centrality=non_centrality), 'k,')
    plt.plot([], [], 'k--', label=r'$C^{-1}_{\nu}(x;\lambda)$')
    plt.plot([], [], 'k-', label=r'$\tilde{C}^{-1}_{\nu}(x;\lambda)$')
    plt.ylim(0, 50)
    plt.yticks([i for i in range(0, 51, 10)])
    plt.xticks([0, 1])
    plt.xlabel(r'$x$')
    plt.legend(frameon=False)
    if save_figure:
        plt.savefig('non_central_chi_squared_linear_approximation.pdf', format='pdf', bbox_inches='tight', transparent=True)


def print_speed_up_and_efficiency():
    for V, c in [[2 ** -1, 1.0 / 9.0], [2 ** -13, 1.0 / 6.0], [2 ** -14, 1.0 / 7.0], [2 ** -25, 1.0 / 5.0]]:
        C = 1.0 + c
        e = (1.0 + np.sqrt(V * C / c)) ** 2
        s = c * e
        m = np.sqrt(s/c)
        M = np.sqrt(s * V /C)
        print(1.0 / s, 100.0 / e, m, 1.0/M)
