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
import json


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


def plot_piecewise_constant_approximation(savefig=False, plot_from_json=True):
    if plot_from_json:
        with open('piecewise_constant_gaussian_approximation.json', "r") as input_file:
            results = json.load(input_file)
        u, exact, approximation = results['uniforms'], results['exact'], results['approximate']
    else:
        u = np.linspace(0, 1, 1000)[1:-1]
        norm_inv_approx = construct_piecewise_constant_approximation(norm.ppf, 8)
        exact = norm_inv(u)
        approximation = norm_inv_approx(u)
    plt.clf()
    plt.plot(u, exact, 'k--', label=r'$\Phi^{-1}(x)$')
    plt.plot(u, approximation, 'k,', label=r'__nolegend__')
    plt.plot([], [], 'k-', label=r'$Q(x)$')
    plt.xlabel(r"$x$")
    plt.xticks([0, 1])
    plt.yticks([-3, 0, 3])
    plt.ylim(-3, 3)
    plt.legend(frameon=False)
    if savefig:
        plt.savefig('piecewise_constant_gaussian_approximation.pdf', format='pdf', bbox_inches='tight', transparent=True)
        if not plot_from_json:
            with open('piecewise_constant_gaussian_approximation.json', "w") as output_file:
                output_file.write(json.dumps({'uniforms': u.tolist(), 'exact': norm_inv(u).tolist(), 'approximate': norm_inv_approx(u).tolist()}, indent=4))


def plot_piecewise_constant_error(savefig=False, plot_from_json=True):
    if plot_from_json:
        with open('piecewise_constant_gaussian_approximation_error.json', "r") as input_file:
            results = json.load(input_file)
        for p in results:
            for data_type in ['data', 'bound']:
                results[p][data_type] = {float(k): float(v) for k, v in results[p][data_type].items()}
    else:
        results = {1 << i: {'data': {}, 'bound': {}} for i in range(1, 4)}
        for n in [1 << i for i in range(11)]:
            norm_inv_approx = construct_piecewise_constant_approximation(norm.ppf, n)
            discontinuities = np.linspace(0, 1, n + 1)
            for p in results:
                p_norm = integrate(lambda u: (norm_inv(u) - norm_inv_approx(u)) ** p, 0, 1, points=discontinuities, limit=50 + 10 * n)[0] ** (1.0 / p)
                results[p]['data'][n] = p_norm
        for p in results:
            n, p_norm = zip(*results[p]['data'].items())
            q = np.linspace(2, np.log2(n[-1]), 100)  # For the analytic bound from Giles
            x = 2.0 ** q
            y = 2.0 ** (-q / p) * q ** -0.5
            y = y / y[-1] * p_norm[-1]  # Rescaled
            results[p]['bound'] = {k: v for k, v in zip(x, y)}

    plt.clf()
    markers = (i for i in {'o', 's', 'v'})
    for p in results:
        n, p_norm = zip(*results[p]['data'].items())
        n_bound, p_bound = zip(*results[p]['bound'].items())
        marker = next(markers)
        plt.plot(n, p_norm, 'k{}'.format(marker), label=r'$p = {}$'.format(p))
        plt.plot(n, p_norm, 'k:', label=r'__nolegend__')
        plt.plot(n_bound, p_bound, 'k--', label='__nolegend__')
    plt.plot([], [], 'k--', label=r'$O(2^{-q/p} q^{-1/2})$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$\lVert Z - \widetilde{Z}\rVert_p$')
    plt.xlabel('Intervals')
    plt.legend(frameon=False, handlelength=1, borderaxespad=0)
    if savefig:
        plt.savefig('piecewise_constant_gaussian_approximation_error.pdf', format='pdf', bbox_inches='tight', transparent=True)
        if not plot_from_json:
            with open('piecewise_constant_gaussian_approximation_error.json', "w") as output_file:
                output_file.write(json.dumps(results, indent=4))


def plot_piecewise_linear_gaussian_approximation(savefig=False, plot_from_json=True):
    if plot_from_json:
        with open('piecewise_linear_gaussian_approximation.json', "r") as input_file:
            results = json.load(input_file)
        u, exact, approximate = results['uniforms'], results['exact'], results['approximate']
    else:
        u = np.linspace(0, 1, 1000)[1:-1]
        norm_inv_approx = construct_symmetric_piecewise_polynomial_approximation(norm.ppf, n_intervals=5, polynomial_order=1)
        exact, approximate = norm_inv(u), norm_inv_approx(u)
    plt.clf()
    plt.plot(u, exact, 'k--', label=r'$\Phi^{-1}(x)$')
    plt.plot(u, approximate, 'k,', label=r'__nolegend__')
    plt.plot([], [], 'k-', label=r'$D(x)$')
    plt.xlabel(r"$x$")
    plt.xticks([0, 1])
    plt.yticks([-3, 0, 3])
    plt.ylim(-3, 3)
    plt.legend(frameon=False)
    if savefig:
        plt.savefig('piecewise_linear_gaussian_approximation.pdf', format='pdf', bbox_inches='tight', transparent=True)
        if not plot_from_json:
            with open('piecewise_linear_gaussian_approximation.json', "w") as output_file:
                output_file.write(json.dumps({'uniforms': u.tolist(), 'exact': norm_inv(u).tolist(), 'approximate': norm_inv_approx(u).tolist()}, indent=4))


def plot_piecewise_linear_gaussian_approximation_error(savefig=False, plot_from_json=True):
    if plot_from_json:
        with open('piecewise_linear_gaussian_approximation_error.json', "r") as input_file:
            results = json.load(input_file)
        results = {k: {int(x): float(y) for x, y in v.items()} for k, v in results.items()}
    else:
        polynomial_orders = range(6)
        interval_sizes = [2, 4, 8, 16]
        results = {s: {} for s in interval_sizes}
        for n_intervals in results:
            for polynomial_order in polynomial_orders:
                approximate_inverse_gaussian_cdf = construct_symmetric_piecewise_polynomial_approximation(norm.ppf, n_intervals + 1, polynomial_order)  # +1 as we have the 0 interval which is measure 0.
                discontinuities = [0.5 ** (i + 2) for i in range(n_intervals)]  # Makes the numerical integration involved in the RMSE easier.
                rmse = integrate(lambda u: 2.0 * (norm.ppf(u) - approximate_inverse_gaussian_cdf(u)) ** 2, 0, 0.5, points=discontinuities)[0] ** 0.5
                results[n_intervals][polynomial_order] = rmse

    polynomial_orders = sorted(list(results.values())[0].keys())
    plt.clf()
    for n_intervals in results:
        poly_orders, rmse = zip(*results[n_intervals].items())
        plt.plot(poly_orders, rmse, 'ko:', label='__nolengend__')
        plt.plot([], [], 'ko', label=n_intervals)
        plt.gca().text(poly_orders[-1] + 0.2, rmse[-1], str(n_intervals), va='center')
    plt.yscale('log')
    plt.ylabel(r'$\lVert Z - \widetilde{Z}\rVert_2$')
    plt.xlabel('Polynomial order')
    plt.xlim(None, 5.7)
    plt.ylim(1e-4, 1e0)
    plt.xticks(polynomial_orders)
    if savefig:
        plt.savefig('piecewise_linear_gaussian_approximation_error.pdf', format='pdf', bbox_inches='tight', transparent=True)
        if not plot_from_json:
            with open('piecewise_linear_gaussian_approximation_error.json', "w") as output_file:
                output_file.write(json.dumps(results, indent=4))


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


def produce_cir_paths_with_only_gaussians(dt, approx=None, **kwargs):
    """
    Perform path simulations of a geometric Brownian motion.
    :param dt: Float. (Fraction of time).
    :param method: Str.
    :param approx: List.
    :return: List. [x_fine_exact, x_coarse_exact, x_fine_approx, x_coarse_approx]
    """
    assert isinstance(dt, float) and np.isfinite(dt) and dt > 0 and (1.0 / dt).is_integer()
    assert approx is not None
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

    # x_0, T, dt, t_fine, t_coarse, sqrt_t_fine, w_coarse_exact, w_coarse_approx = [mpf(i) for i in [x_0, T, dt, t_fine, t_coarse, sqrt_t_fine, w_coarse_exact, w_coarse_approx]]
    # fabs = mp.fabs

    path_update = lambda x, w, t: x + kappa * (theta - x) * t + sigma * np.sqrt(np.fabs(x)) * w

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


def plot_variance_reduction_geometric_brownian_motion(savefig=False, plot_from_json=True):
    methods = ['euler_maruyama', 'milstein']
    if plot_from_json:
        results = {}
        for method in methods:
            with open('variance_reduction_{}_scheme.json'.format(method), "r") as input_file:
                results[method] = json.load(input_file)
            results[method] = {k: {float(x): y for x, y in v.items()} for k, v in results[method].items()}
    else:
        deltas = [2.0 ** -i for i in range(1, 7)]
        inverse_norm = norm.ppf
        piecewise_constant = construct_piecewise_constant_approximation(inverse_norm, n_intervals=1024)
        piecewise_linear = construct_symmetric_piecewise_polynomial_approximation(inverse_norm, n_intervals=16, polynomial_order=1)
        piecewise_cubic = construct_symmetric_piecewise_polynomial_approximation(inverse_norm, n_intervals=16, polynomial_order=3)
        approximations = {'constant': piecewise_constant, 'linear': piecewise_linear, 'cubic': piecewise_cubic, 'rademacher': rademacher_approximation}

        results = {method: {term: {} for term in ['original'] + list(approximations.keys())} for method in methods}  # Store the values of delta and the associated data.
        time_per_level = 2.0
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
                        [mean, std] = [float(i) for i in [mean, std]]
                        results[method][name][dt] = [mean, std]

    markers = {'original': 'd', 'constant': 'o', 'linear': 'v', 'cubic': 's', 'rademacher': 'x'}
    deltas = list(list(list(results.items())[0][1].items())[0][1].keys())
    for method in results:
        plt.clf()
        for approx_name in results[method]:
            x, y = zip(*results[method][approx_name].items())
            y, y_std = list(zip(*y))
            y_error = 1 * np.array(y_std)
            plt.errorbar(x, y, y_error, None, 'k{}:'.format(markers[approx_name]))
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.xlabel(r'Fine time increment $\delta^{\mathrm{f}}$')
        plt.ylabel('Variance')
        y_min_base_2 = 50
        plt.ylim(2 ** -y_min_base_2, 2 ** -10)
        plt.yticks([2 ** -i for i in range(10, y_min_base_2 + 1, 10)])
        plt.xticks(deltas)
        if savefig:
            plt.savefig('variance_reduction_{}_scheme.pdf'.format(method), format='pdf', bbox_inches='tight', transparent=True)
            if not plot_from_json:
                with open('variance_reduction_{}_scheme.json'.format(method), "w") as output_file:
                    output_file.write(json.dumps(results[method], indent=4))


def plot_variance_reduction_cir_with_only_approx_gaussian_mlmc(savefig=False, plot_from_json=True):

    if plot_from_json:
        with open('variance_reduction_cir_with_only_approx_gaussian_mlmc.json', "r") as input_file:
            results = json.load(input_file)
    else:
        deltas = [2.0 ** -i for i in range(1, 9)]
        inverse_norm = norm.ppf
        piecewise_constant = construct_piecewise_constant_approximation(inverse_norm, n_intervals=1024)
        piecewise_linear = construct_symmetric_piecewise_polynomial_approximation(inverse_norm, n_intervals=16, polynomial_order=1)
        approximations = {'constant': piecewise_constant, 'linear': piecewise_linear}
        params = {'kappa': 0.5, 'theta': 1.0, 'sigma': 1.0}
        results = {term: {} for term in ['original'] + list(approximations.keys())}
        time_per_level = 60.0
        paths_min = 64
        for approx_name, approx in approximations.items():
            for dt in deltas:
                _, elapsed_time_per_path = time_function(produce_cir_paths_with_only_gaussians)(dt, approx, **params)
                paths_required = int(time_per_level / elapsed_time_per_path)
                if paths_required < paths_min:
                    print("More time required for {} with dt={}".format(approx_name, dt))
                    break

                originals, corrections = [[None for i in range(paths_required)] for j in range(2)]
                for path in range(paths_required):
                    x_fine_exact, x_coarse_exact, x_fine_approx, x_coarse_approx = produce_cir_paths_with_only_gaussians(dt, approx, **params)
                    originals[path] = x_fine_exact - x_coarse_exact
                    corrections[path] = min((x_fine_exact - x_coarse_exact) - (x_fine_approx - x_coarse_approx), (x_fine_exact - x_fine_approx) - (x_coarse_exact - x_coarse_approx), sum([x_fine_exact, -x_coarse_exact, -x_fine_approx, x_coarse_approx]))  # might need revising for near machine precision.
                originals, corrections = [[j ** 2 for j in i] for i in [originals, corrections]]
                for name, values in [['original', originals], [approx_name, corrections]]:
                    mean = np.mean(values)
                    std = np.std(values) / (len(values) ** 0.5)
                    [mean, std] = [float(i) for i in [mean, std]]
                    results[name][dt] = [mean, std]

    markers = {'original': 'd', 'constant': 'o', 'linear': 'v', 'cubic': 's', 'rademacher': 'x'}
    deltas = list(list(results.items())[0][1].keys())
    levels = [int(i) for i in np.log2(1.0/np.array([float(i) for i in deltas]))]

    plt.clf()
    ls = {'original': (None, None), 'constant': (15, 15), 'linear': (10, 3, 4, 3)}
    leg = {'original': 'baseline', 'constant': 'constant', 'linear': 'dyadic'}
    for approx_name in results:
        x, y = zip(*results[approx_name].items())
        x = [float(i) for i in x]
        l = [int(i) for i in np.log2(1.0/np.array(x))]
        y, y_std = list(zip(*y))
        y_error = 1 * np.array(y_std)
        plt.errorbar(l, y, y_error, None, 'ko-', dashes=ls[approx_name], label=leg[approx_name])
    plt.yscale('log', base=10)
    plt.xlabel(r'level $\ell$')
    plt.ylabel('Variance')
    plt.ylim(1e-8, 1e1)
    plt.xlim(0, None)
    plt.xticks(levels)
    plt.legend(frameon=False, handlelength=4)
    if savefig:
        plt.savefig('variance_reduction_cir_with_only_approx_gaussian_mlmc.pdf', format='pdf', bbox_inches='tight', transparent=True)
        if not plot_from_json:
            with open('variance_reduction_cir_with_only_approx_gaussian_mlmc.json', "w") as output_file:
                output_file.write(json.dumps(results, indent=4))


def inverse_non_central_chi_squared_abdel_aty(u, df, nc):
    """The approximation from Abdel-Aty, cf: https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution#Approximation_(including_for_quantiles)"""
    k = df
    l = nc
    f = (k + l) ** 2 / (k + 2.0*l)
    x = norm.ppf(u)
    x *= np.sqrt(2.0/(9.0 * f))
    x += 1.0 - 2.0/(9.0 * f)
    x = x ** 3
    x *= (k + l)
    return x

def inverse_non_central_chi_squared_sankaran(u, df, nc):
    """The approximation from Sankaran, cf: https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution#Approximation_(including_for_quantiles)"""
    k = df
    l = nc
    h = 1.0 - (2.0/3.0) * (k + l) * (k + 3.0 * l) / ((k + 2.0 * l)**2)
    p = (k + 2.0*l) / ((k + l)**2)
    m = (h - 1.0) * (1 - 3.0*h)
    x = norm.ppf(u)
    x *= h * np.sqrt(2.0 * p) * (1.0 + 0.5 * m * p)
    x += 1.0 + h * p * (h - 1.0 + 0.5 * (2.0 - h) * m * p)
    x = x ** (1.0/h)
    x *= (k + l)
    return x



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

    # For the approximations by Abdel-Aty and Sankaran
    approxes = {'Abdel-Aty': inverse_non_central_chi_squared_abdel_aty, 'Sankaran': inverse_non_central_chi_squared_sankaran}
    results = {approx: {nu: {} for nu in nus} for approx in approxes.keys()}
    for name in results.keys():
        ncx2_approx = approxes[name]
        for nu in progressbar(nus):
            for l in lambdas:
                limits=[50,10,1]
                for limit in limits:
                    rmse = integrate(lambda u: (ncx2.ppf(u, df=nu, nc=l) - ncx2_approx(u, df=nu, nc=l)) ** 2, 0, 1, limit=limit)[0] ** 0.5
                    if not np.isnan(rmse):
                        break
                results[name][nu][l] = rmse

    for name, result in results.items():
        df = pd.DataFrame(result)
        df.index = df.index.rename('lambda')
        df.columns = df.columns.rename('nu')
        print(name, df.min().min(), df.max().max())
        print(round(df, 3))
        print('\n')
        print(round(df, 3).apply(lambda x: ' & '.join([str(i) for i in list(x)]) + r' \\', axis=1))
        print('\n' * 3)


def produce_cox_ingersoll_ross_paths_by_approx_euler_maruyama(dt, gaussian_approximations=None, **kwargs):
    assert isinstance(dt, float) and np.isfinite(dt) and dt > 0 and (1.0 / dt).is_integer()
    assert gaussian_approximations is not None
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

    x_euler_maruyama = x_0
    x_approximations = [x_0] * len(gaussian_approximations)

    n_increments = int(1.0 / dt)

    for n in range(n_increments):
        u = np.random.uniform()
        z = norm.ppf(u)
        z_approxes = [approx(u) for approx in gaussian_approximations]
        dw = sqrt_t * z
        dw_approxes = [sqrt_t * z in z_approxes]
        x_euler_maruyama = euler_maruyama_update(x_euler_maruyama, dw, dt)
        x_approximations = [euler_maruyama_update(x_euler_maruyama, dw_approx, dt) for dw_approx in dw_approxes]

    return [x_euler_maruyama, *x_approximations]

def produce_cox_ingersoll_ross_paths(dt, approximations=None, full_path=False, **kwargs):
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

    if full_path:
        paths = []
        paths.append([x_euler_maruyama, x_exact, *x_approximations])

    n_increments = int(1.0 / dt)

    for n in range(n_increments):
        u = np.random.uniform()
        z = norm.ppf(u)
        dw = sqrt_t * z
        x_euler_maruyama = euler_maruyama_update(x_euler_maruyama, dw, dt)
        x_exact = exact_update(u, x_exact)
        x_approximations = [approximate_update(u, x_approximate, approx) for approx, x_approximate in zip(approximations, x_approximations)]
        if full_path:
            paths.append([x_euler_maruyama, x_exact, *x_approximations])

    if full_path:
        x_euler_maruyama, x_exact, *x_approximations = list(zip(*paths))

    return [x_euler_maruyama, x_exact, *x_approximations]


def plot_variance_reduction_cir_process(savefig=False, plot_from_json=True):
    poly_orders = {'linear': 1, 'cubic': 3}
    poly_markers = (i for i in ['s', 'd'])
    markers = {**{'exact': 'o', 'euler_maruyama': 'v'}, **{k: next(poly_markers) for k in poly_orders}}
    if plot_from_json:
        with open('variance_reduction_cir_process.json', "r") as input_file:
            results = json.load(input_file)
        results = {k: {float(x): y for x, y in v.items()} for k, v in results.items()}
    else:
        deltas = [0.5 ** i for i in range(8)]
        params = {'kappa': 0.5, 'theta': 1.0, 'sigma': 1.0}
        nu = 4.0 * params['kappa'] * params['theta'] / (params['sigma'] ** 2)
        approximations = [construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation(dof=nu, polynomial_order=poly_order) for poly_order in [1, 3]]

        results = {k: {} for k in ['exact', 'euler_maruyama'] + list(poly_orders.keys())}
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
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xticks(x)
    plt.ylim(2 ** -25, 2 ** 2)
    plt.yticks([2 ** -i for i in range(0, 30, 5)])
    plt.xlabel(r'Time increment $\delta$')
    plt.ylabel('Variance')
    if savefig:
        plt.savefig('variance_reduction_cir_process.pdf', format='pdf', bbox_inches='tight', transparent=True)
        if not plot_from_json:
            with open('variance_reduction_cir_process.json', "w") as output_file:
                output_file.write(json.dumps(results, indent=4))

def plot_variance_reduction_cir_process_asian_option(savefig=False, plot_from_json=True):
    poly_orders = {'linear': 1, 'cubic': 3}
    poly_markers = (i for i in ['s', 'd'])
    markers = {**{'exact': 'o', 'euler_maruyama': 'v'}, **{k: next(poly_markers) for k in poly_orders}}
    full_paths = True
    if plot_from_json:
        with open('variance_reduction_cir_process_asian_option.json', "r") as input_file:
            results = json.load(input_file)
        results = {k: {float(x): y for x, y in v.items()} for k, v in results.items()}
    else:
        deltas = [0.5 ** i for i in range(8)]
        params = {'kappa': 0.5, 'theta': 1.0, 'sigma': 1.0}
        nu = 4.0 * params['kappa'] * params['theta'] / (params['sigma'] ** 2)
        approximations = [construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation(dof=nu, polynomial_order=poly_order) for poly_order in [1, 3]]

        results = {k: {} for k in ['exact', 'euler_maruyama'] + list(poly_orders.keys())}
        time_per_level = 5.0
        paths_min = 64
        for dt in progressbar(deltas):
            _, elapsed_time_per_path = time_function(produce_cox_ingersoll_ross_paths)(dt, approximations, full_paths, **params)
            paths_required = int(time_per_level / elapsed_time_per_path)
            if paths_required < paths_min:
                print("More time required for dt={}".format(dt))
                break
            exacts, euler_maruyamas, linears, cubics = [[None for i in range(paths_required)] for j in range(4)]
            for path in range(paths_required):
                x_euler_maruyama, x_exact, x_linear, x_cubic = produce_cox_ingersoll_ross_paths(dt, approximations, full_paths, **params)
                x_euler_maruyama, x_exact, x_linear, x_cubic = [np.mean(i) for i in [x_euler_maruyama, x_exact, x_linear, x_cubic]]  # The arithmetic mean
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
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xticks(x)
    plt.ylim(2 ** -27, 2 ** 2)
    plt.yticks([2 ** -i for i in range(0, 30, 5)])
    plt.xlabel(r'Time increment $\delta$')
    plt.ylabel('Variance')
    if savefig:
        plt.savefig('variance_reduction_cir_process_asian_option.pdf', format='pdf', bbox_inches='tight', transparent=True)
        if not plot_from_json:
            with open('variance_reduction_cir_process_asian_option.json', "w") as output_file:
                output_file.write(json.dumps(results, indent=4))


def plot_non_central_chi_squared_polynomial_approximation(savefig=False, plot_from_json=True):
    if plot_from_json:
        with open('non_central_chi_squared_linear_approximation.json', "r") as input_file:
            results = json.load(input_file)
        results = {k: {x: {float(u): w for u, w in y.items()} for x, y in v.items()} for k, v in results.items()}
    else:
        dof = 1.0
        ncx2_approx = construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation(dof, n_intervals=4 + 1)
        u = np.concatenate([np.linspace(0.0, 1.0, 1000)[:-1], np.logspace(-10, -1, 100), 1.0 - np.logspace(-10, -1, 100)])
        u.sort()
        non_centralities = [1.0, 10.0, 20.0]
        results = {non_centrality: {} for non_centrality in non_centralities}
        for non_centrality in results:
            exact, approximate = ncx2.ppf(u, df=dof, nc=non_centrality), ncx2_approx(u, non_centrality=non_centrality)
            results[non_centrality]['exact'] = {x: y for x, y in zip(u, exact)}
            results[non_centrality]['approximate'] = {x: y for x, y in zip(u, approximate)}
            _u = u[1:-1] # The end points can be singular, so we avoid these.
            abdel_aty = inverse_non_central_chi_squared_abdel_aty(_u, df=dof, nc=non_centrality)
            sankaran = inverse_non_central_chi_squared_sankaran(_u, df=dof, nc=non_centrality)
            results[non_centrality]['abdel_aty'] = {x: y for x, y in zip(_u, abdel_aty)}
            results[non_centrality]['sankaran'] = {x: y for x, y in zip(_u, sankaran)}


    plt.clf()
    for non_centrality in results:
        exact, approximate = results[non_centrality]['exact'], results[non_centrality]['approximate']
        plt.plot(*zip(*exact.items()), 'k--')
        plt.plot(*zip(*approximate.items()), 'k,')
        abdel_aty, sankaran = results[non_centrality]['abdel_aty'], results[non_centrality]['sankaran']
        # plt.plot(*zip(*abdel_aty.items()), 'r,')
        # plt.plot(*zip(*abdel_aty.items()), 'b,')
    plt.plot([], [], 'k--', label=r'$C^{-1}_{\nu}(x;\lambda)$')
    plt.plot([], [], 'k-', label=r'$\widetilde{C}^{-1}_{\nu}(x;\lambda)$')
    plt.ylim(0, 50)
    plt.yticks([i for i in range(0, 51, 10)])
    plt.xticks([0, 1])
    plt.xlabel(r'$x$')
    plt.legend(frameon=False)
    if savefig:
        plt.savefig('non_central_chi_squared_linear_approximation.pdf', format='pdf', bbox_inches='tight', transparent=True)
        if not plot_from_json:
            with open('non_central_chi_squared_linear_approximation.json', "w") as output_file:
                output_file.write(json.dumps(results, indent=4))


def print_speed_up_and_efficiencies(variances_reductions, cost_reductions):
    for V, c in zip(variances_reductions, cost_reductions):
        c = 1.0 / c
        C = 1.0 + c
        e = (1.0 + np.sqrt(V * C / c)) ** 2
        s = c * e
        m = np.sqrt(s / c)
        M = np.sqrt(s * V / C)
        print(1.0 / s, 100.0 / e, m, 1.0 / M)


def print_speed_up_and_efficiencies_non_central_chi_squared():
    variances_reductions = [2 ** -15, 2 ** -25]
    cost_reductions = [300.0, 300.0]
    print_speed_up_and_efficiencies(variances_reductions, cost_reductions)


def print_speed_up_and_efficiencies_gaussian():
    variances_reductions = [2 ** -1, 2 ** -13, 2 ** -14, 2 ** -25]
    cost_reductions = [9.0, 6.0, 7.0, 5.0]
    print_speed_up_and_efficiencies(variances_reductions, cost_reductions)


if __name__ == '__main__':
    plot_params = dict(savefig=True, plot_from_json=True)
    # plot_params = dict(savefig=True, plot_from_json=False)
    # plot_params = dict(savefig=False, plot_from_json=True)
    # plot_params = dict(savefig=False, plot_from_json=False)
    plot_piecewise_constant_approximation(**plot_params)
    plot_piecewise_constant_error(**plot_params)
    plot_piecewise_linear_gaussian_approximation(**plot_params)
    plot_piecewise_linear_gaussian_approximation_error(**plot_params)
    plot_variance_reduction_geometric_brownian_motion(**plot_params)
    plot_variance_reduction_cir_with_only_approx_gaussian_mlmc(**plot_params)
    plot_variance_reduction_cir_process(**plot_params)
    plot_variance_reduction_cir_process_asian_option(**plot_params)
    plot_non_central_chi_squared_polynomial_approximation(**plot_params)