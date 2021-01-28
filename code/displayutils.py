import sys
import matplotlib.pyplot as plt
import numpy as np

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
    
def errorfill(x, y, yerr, color=None, alpha_fill=0.1, ax=None, linestyle=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
        ind_min = np.where(ymin<0)
        ind_max = np.where(ymax>1)
        ymax[ind_max] = 1.0
        ymin[ind_min] = 0.0
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, linestyle=linestyle)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)    
