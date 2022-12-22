"""
Author:

    Oliver Sheridan-Methven, October 2020.

Description:

    The configurations for the plots.
"""

import matplotlib as mpl
figure_size = (2.8, 2)
rc_fonts = {
    "font.family": "serif",
    "font.size": 9,
    'figure.figsize': figure_size,
    'lines.linewidth': 0.5,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'lines.markersize': 3,
    "text.usetex": True,
    # 'text.latex.preview': True, # Not in newer python.
}
style = 'arxiv'
style = 'acm'
preamble = ''


style = 'arxiv'
if style == 'arxiv':
    rc_fonts_extras = {"font.serif": "computer modern roman"}
    preamble = r'\usepackage{amsmath,amssymb,bbm,bm,physics,fixcmex}'
elif style == 'acm':
    preamble = r"""
    \usepackage{amsmath,amssymb,bbm,bm,physics} 
    \usepackage{libertine} 
    \usepackage[libertine]{newtxmath}
    """
else:
    raise NotImplementedError

rc_fonts_extras = {}


rc_fonts = {**rc_fonts, **rc_fonts_extras}
mpl.rcParams.update(rc_fonts)
import matplotlib.pylab as plt
plt.rc('text.latex', preamble=preamble) # Multi-line preambles need to be set here and not in rc_fonts_extras annoyingly.
plt.ion()

