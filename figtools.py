from __future__ import division

import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#=========================================================================================
# Default settings

# Tick direction out
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'

# Axis labels
mpl.rcParams['axes.labelsize'] = 7

# legend font size
mpl.rcParams['legend.fontsize'] = 7

# font size
mpl.rcParams['font.size'] = 7

# font family
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']

# Line width
mpl.rcParams['lines.linewidth'] = 1

# use sans-serif for math
mpl.rcParams['mathtext.fontset'] = 'stixsans'

#=========================================================================================
# MySubplot

class MySubplot(object):
    labelpadx = 1.5
    labelpady = 2

    def __init__(self, fig, rect):
        self.ax    = fig.add_axes(rect)
        self.xaxis = self.ax.xaxis
        self.yaxis = self.ax.yaxis

    #-------------------------------------------------------------------------------------
    # Plot frame

    def set_frame_thickness(self, thickness, color=None):
        for v in self.ax.spines.values():
            v.set_linewidth(thickness)

        self.xaxis.set_tick_params(width=thickness)
        self.yaxis.set_tick_params(width=thickness)

        if color is not None:
            for s in self.ax.spines.values():
                s.set_color(color)

    def set_tick_params(self, ticksize, ticklabelsize, pad=None):
        self.xaxis.set_tick_params(size=ticksize, labelsize=ticklabelsize)
        self.yaxis.set_tick_params(size=ticksize, labelsize=ticklabelsize)

        if pad is not None:
            self.xaxis.set_tick_params(pad=pad)
            self.yaxis.set_tick_params(pad=pad)

    def bottomleft_axes(self, thickness, ticksize, ticklabelsize, pad=None):
        for s in ['top', 'right']:
            self.ax.spines[s].set_visible(False)

        self.xaxis.tick_bottom()
        self.yaxis.tick_left()

        self.set_frame_thickness(thickness)
        self.set_tick_params(ticksize, ticklabelsize, pad)

    #-------------------------------------------------------------------------------------
    # Improved plotting interface

    def plot(self, *args, **kargs):
        if 'clip_on' not in kargs:
            kargs['clip_on'] = False
        return self.ax.plot(*args, **kargs)

    def xlabel(self, *args, **kargs):
        if 'labelpad' not in kargs:
            kargs['labelpad'] = MySubplot.labelpadx
        return self.ax.set_xlabel(*args, **kargs)

    def ylabel(self, *args, **kargs):
        if 'labelpad' not in kargs:
            kargs['labelpad'] = MySubplot.labelpadx
        return self.ax.set_ylabel(*args, **kargs)

    def xlim(self, *args):
        return self.ax.set_xlim(*args)

    def ylim(self, *args):
        return self.ax.set_ylim(*args)

    def set_equal(self):
        return self.ax.set_aspect('equal')

    def bar(self,*args, **kargs):
        return self.ax.bar(*args,**kargs)


    #-------------------------------------------------------------------------------------
    # Axis limits

    def lim(self, axis, data, margin=0.05, relative=True, lower=None, upper=None):
        xmin, xmax = np.min(data), np.max(data)
        if lower is not None:
            xmin = lower
        if upper is not None:
            xmax = upper

        if relative:
            dx = margin*(xmax - xmin)
        else:
            dx = margin

        xlims = xmin-dx, xmax+dx
        if lower is not None:
            xlims = lower, xlims[1]
        if upper is not None:
            xlims = xlims[0], upper

        if axis == 'x':
            self.xlim(*xlims)
        else:
            self.ylim(*xlims)

        return xlims
    
    #-------------------------------------------------------------------------------------
    # Annotation

    def legend(self, *args, **kargs):
        if 'bbox_transform' not in kargs:
            kargs['bbox_transform'] = self.ax.transAxes
        if 'frameon' not in kargs:
            kargs['frameon'] = False
        if 'numpoints' not in kargs:
            kargs['numpoints'] = 1

        return self.ax.legend(*args, **kargs)
    
#=========================================================================================
# MyFigure

class MyFigure(object):
    defaultwidth     = 6.5
    defaultheight    = 5.5
    defaultrowheight = 3.5

    default_rect = [0.12, 0.13, 0.8, 0.8]
    default_bottomleft_params = {'thickness':     0.75,
                                 'ticksize':      2,
                                 'ticklabelsize': 7}
    
    def __init__(self, figsize=None, specs=None):
        self.plots = []

        if figsize is None:
            w = MyFigure.defaultwidth
            h = MyFigure.defaultheight
        else:
            w = figsize[0]
            h = figsize[1]

        self.fig = plt.figure(figsize=(w, h))

    def addplot(self, rect=None, bottomleftaxes=True):
        if rect is None:
            rect = MyFigure.default_rect
        
        plot = MySubplot(self.fig, rect)
        self.plots.append(plot)

        if bottomleftaxes:
            plot.bottomleft_axes(**self.bottomleft_params())

        return plot

    def bottomleft_params(self):
        return MyFigure.default_bottomleft_params

    def save(self, filename, path='', ext='pdf'):
        f = path + filename + '.' + ext
        plt.savefig(f, transparent=True)
        print("* MyFigure saved as {}.".format(f))

    def close(self):
        plt.close()
