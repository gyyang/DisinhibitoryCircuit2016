'''2014-10-14
Fitting curves, approximating spiking network with rates
'''

from __future__ import division
import datetime
import copy
import numpy as np
import pickle
import scipy.optimize
from figtools import MyFigure, MySubplot
from MultiCompartmentalModel import read_params

#version = str(datetime.date.today()) + '_1'


class Study():
    def __init__(self,figpath='figure/', datapath='data/', version=None,
                 paramsfile='parameters.text',eqsfile='equations.txt'):
        self.figpath = figpath
        self.datapath = datapath
        if version is None:
            version = '_'+str(datetime.date.today())

        self.version = version

        self.paramsfile = paramsfile
        self.eqsfile = eqsfile

        self.params = read_params(paramsfile)


    def obj_func_DendVvsgErI(self,p,result,rich_return=False):
        n_curve = len(result['vdend_mean_list'])
        sqe = 0
        y_list = list()
        for i in xrange(n_curve):
            ydata = result['vdend_mean_list'][i]*1000

            inh = result['Inh'][i]
            #mid_point = p[0] + inh*p[1]
            mid_point = (4+inh)*p[0]
            spread = p[1]*np.exp(inh/p[2])
            y = 30*(1+np.tanh((result['Exc']-mid_point)/spread)) + p[3] -70
            sqe += sum((y-ydata)**2)

            if rich_return:
                y_list.append(y)
        if rich_return:
            return y_list, sqe
        else:
            return sqe

    def fit_DendVvsgErI(self):
        '''
        fit the dendritic voltage as a function of excitatory input and inhibitory rate
        '''
        with open(self.datapath+'DendVvsgErI'+self.version,'rb') as f:
            result = pickle.load(f)

        x0 = [5,10,7,1]
        bnds = ((1,20),(1,20),(1,20),(0,3))

        params = copy.copy(self.params)
        for key in result['params']:
            params[key] = result['params'][key] # overwrite the old value


        temp = params['pre_rate']*params['tauNMDARise']*params['tauNMDADecay']*params['alphaNMDA']
        print params['tauNMDARise']*params['tauNMDADecay']*params['alphaNMDA']
        # 0.06 is obtained from tau_rise = 2ms, tau_decay=100ms, alpha=0.3 /ms
        s_mean = temp/(1+temp) # mean gating variable
        result['Exc'] = result['weights']*params['gNMDA']*1e9*params['num_input']*s_mean
        # unit-less weights * synaptic conductance 2.5nS * 15 synapses * mean gating variable for 30Hz

        result['Inh'] = result['dend_inh_rates'] *params['tauGABA']*params['gGABA']*1e9
        # Total Rate * synaptic conductance * tau_GABA

        res = scipy.optimize.minimize(lambda p: self.obj_func_DendVvsgErI(p,result),
                                      x0,bounds=bnds,method='SLSQP',
                                      options={'maxiter':100, 'disp':True})
        print res.x
        p = res.x

        xdata = result['Exc']
        n_curve = len(result['vdend_mean_list'])

        fig=MyFigure(figsize=(2,1.5))
        pl = fig.addplot(rect=[0.25,0.25,0.7,0.7])

        for i in xrange(n_curve):
            ydata = result['vdend_mean_list'][i]*1000
            ax_data, = pl.plot(xdata,ydata,color=np.array([55,126,184])/255)

        ymodel_list, sqe = self.obj_func_DendVvsgErI(p,result,rich_return=True)
        for i in xrange(n_curve):
            ax_model, = pl.plot(xdata,ymodel_list[i],color='black')


        pl.xlabel(r'$g_{E,\mathrm{tot}}$ (nS)',labelpad = 3)
        pl.ylabel(r'$\overline{V}_D$ (mV)')
        pl.ax.set_xlim((0,50))
        pl.ax.set_xticks([0,25,50])
        pl.ax.set_ylim((-70,-10))
        pl.ax.set_yticks([-70,-20])
        leg = pl.legend([ax_data,ax_model],['Simulation','Fit'],loc=2, bbox_to_anchor=[0, 1.05])
        leg.draw_frame(False)

        fig.save(self.figpath+'DendVvsgErI'+self.version)

    def obj_func_RatevsDendV(self,p,result,rich_return=False):
        xdata = result['clamped_dendvolt_list'] + 70
        ydata = result['fr_list']
        y = p[0]+p[1]*(xdata**p[2])
        sqe = sum((y-ydata)**2)

        if rich_return:
            return y, sqe
        else:
            return sqe

    def fit_RatevsDendV(self):
        '''
        fit the somatic firing rate as a function of mean dendritic voltage
        '''
        with open(self.datapath+'RatevsDendV'+self.version,'rb') as f:
            result = pickle.load(f)

        x0 = [1.5,0.08,2.2]
        bnds = ((0,5),(0,1),(1,3))

        params = copy.copy(self.params)
        for key in result['params']:
            params[key] = result['params'][key] # overwrite the old value

        res = scipy.optimize.minimize(lambda p: self.obj_func_RatevsDendV(p,result),
                                      x0,bounds=bnds,method='SLSQP',
                                      options={'maxiter':100, 'disp':True})
        print res.x
        p = res.x

        xdata = result['clamped_dendvolt_list'] + 70
        ydata = result['fr_list']
        ymodel, _ = self.obj_func_RatevsDendV(p,result,rich_return=True)

        fig=MyFigure(figsize=(2,1.5))
        pl = fig.addplot(rect=[0.25,0.25,0.7,0.7])
        pl.plot(xdata-70,ydata,color=np.array([55,126,184])/255)
        pl.plot(xdata-70,ymodel,color='black')
        pl.ax.set_xticks([-70,-60,-50,-40])
        pl.ax.set_yticks([0,50,100,150])
        pl.xlabel(r'$\langle \overline{V}_D\rangle$ (mV)')
        pl.ylabel('Firing rate (Hz)')
        leg = pl.legend(['Simulation','Fit'],loc=2)
        leg.draw_frame(False)

        fig.save(self.figpath+'RatevsDendV'+self.version)

    def obj_func_RatevsI(self,p,result,rich_return=False):
        xdata = result['clamped_current_list']
        ydata = result['fr_list']
        y = ((xdata+p[0])*(xdata>-p[0])/p[1])**p[2]
        sqe = sum((y-ydata)**2)

        if rich_return:
            return y, sqe
        else:
            return sqe

    def fit_RatevsI(self):
        '''
        fit the somatic firing rate as a function of current injection
        '''
        with open(self.datapath+'RatevsI'+self.version,'rb') as f:
            result = pickle.load(f)

        x0 = [ 174.85951028,   45.16030095,    2.89003711]
        bnds = ((100,500),(10,1000),(0,5))

        res = scipy.optimize.minimize(lambda p: self.obj_func_RatevsI(p,result),
                                      x0,bounds=bnds,method='SLSQP',
                                      options={'maxiter':100, 'disp':True})
        print res.x
        p = res.x

        xdata = result['clamped_current_list']
        ydata = result['fr_list']
        ymodel, _ = self.obj_func_RatevsI(p,result,rich_return=True)

        fig=MyFigure(figsize=(2,1.5))
        pl = fig.addplot(rect=[0.25,0.25,0.7,0.7])
        pl.plot(xdata,ydata,color=np.array([55,126,184])/255)
        pl.plot(xdata,ymodel,color='black')
        pl.ax.set_yticks([0,50,100])
        pl.xlabel(r'$I$ (pA)')
        pl.ylabel('Firing rate (Hz)')
        leg = pl.legend(['Simulation','Fit'],loc=2)
        leg.draw_frame(False)

        fig.save(self.figpath+'RatevsI'+self.version)

