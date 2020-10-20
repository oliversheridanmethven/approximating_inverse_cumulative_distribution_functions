"""
Author:

    Oliver Sheridan-Methven, October 2020.

Description:

    The configurations for the plots.
"""

import matplotlib as mpl
rc_fonts = {
    "font.family": "serif",
    "font.size": 9,
    'figure.figsize': (2.8, 2),
    'lines.linewidth': 0.5,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'lines.markersize': 3,
    "text.usetex": True,
    'text.latex.preview': True,
    'text.latex.preamble': [
        r"""
        \usepackage{amsmath,amssymb,bbm,bm,physics}
        \usepackage{libertine}
        \usepackage[libertine]{newtxmath}
        """],
}
mpl.rcParams.update(rc_fonts)
import matplotlib.pylab as plt
plt.ion()
