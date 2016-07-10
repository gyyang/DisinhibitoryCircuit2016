# -*- coding: utf-8 -*-
"""
Investigate how gating selectivity depends on various parameters

@author: guangyuyang
"""
from __future__ import division
from sys import exc_clear
import time
import os
import re
import numpy.random
import bisect
import scipy as sp
import scipy.optimize
import scipy.misc
import random as pyrand
import pickle
from matplotlib.mlab import PCA
import brian_no_units
from brian import *
from figtools import MyFigure, MySubplot
import MultiCompartmentalModel as MCM

dend_IO = MCM.dend_IO
soma_IO = MCM.soma_IO
soma_fv = MCM.soma_fv

colorMatrix1 = array([[127,205,187],
                    [65,182,196],
                    [29,145,192],
                    [34,94,168],
                    [37,52,148],
                    [8,29,88]])/255

colorMatrix2 = array([[102,194,164],
                    [44,162,95],
                    [0,109,44]])/255
                
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

    def get_proboverlap(self,N,m,k):
        '''
        Choose m elements out of N elements, and then choose m elements independently again
        The probability that there are k overlapping elements
        k \in [0,m]
        :param N: Total number of elements
        :param m: number of elements chosen each time
        :param k: the overlapping number between two times
        :return:
        '''
        if N-m<m-k:
            return 0
        else:
            a = scipy.misc.comb(m,k)
            b = scipy.misc.comb(N-m,m-k)
            c = scipy.misc.comb(N,m)
            p = a*b/c
        return p

    def analytic_evalgating(self,p,no_overlap=False):
        N = p['num_DendEach']
        m = p['num_DendDisinh']

        if ('exc_stim_rate' in p) and (p['exc_stim_rate'] is not None):
            num_syn = 15
            g_exc = p['g_exc']*num_syn
            # gating variable
            s = MCM.meansNMDA(p['exc_stim_rate'])

            # Total conductance input
            exc = s*g_exc # nS
            print exc
        else:
            v_target = p['v_target']
            z = scipy.optimize.minimize_scalar(lambda w: (v_target-dend_IO(w,p['inh_base']))**2,
                                                     bounds=(0,50),method='Bounded')
            exc = z.x


        dend_v = dend_IO(array([exc,exc+exc,exc,0,0]),
                 array([p['inh_base'],p['inh_base'],p['inh_max'],p['inh_max'],p['inh_base']]))
        vdendmean = (m*dend_v[4]+(N-m)*dend_v[3])/N
        rnone = soma_fv(vdendmean)

        vdendmean = (m*dend_v[0]+(N-m)*dend_v[3])/N
        ron = soma_fv(vdendmean)
        rboth = 0
        roff = 0

        if no_overlap:
            i_overlap = 0
            vdendmean = (dend_v[0]*(m-i_overlap)+dend_v[1]*i_overlap
                         +dend_v[2]*(m-i_overlap)+dend_v[3]*(N-2*m+i_overlap))/N
            rboth = soma_fv(vdendmean)

            vdendmean = (dend_v[4]*(m-i_overlap)+dend_v[0]*i_overlap+
                        dend_v[2]*(m-i_overlap) + dend_v[3]*(N-2*m+i_overlap))/N
            roff = soma_fv(vdendmean)
        else:
            for i_overlap in xrange(m+1):
                p_overlap = self.get_proboverlap(N,m,i_overlap)

                vdendmean = (dend_v[0]*(m-i_overlap)+dend_v[1]*i_overlap
                             +dend_v[2]*(m-i_overlap)+dend_v[3]*(N-2*m+i_overlap))/N
                rbothtemp = soma_fv(vdendmean)
                rboth += rbothtemp * p_overlap

                vdendmean = (dend_v[4]*(m-i_overlap)+dend_v[0]*i_overlap+
                            dend_v[2]*(m-i_overlap) + dend_v[3]*(N-2*m+i_overlap))/N
                rofftemp = soma_fv(vdendmean)
                roff += rofftemp * p_overlap


        res = dict()
        res['ron'] = ron
        res['roff'] = roff
        res['rboth'] = rboth
        res['rnone'] = rnone


        dron = ron-rnone
        droff = roff-rnone
        drboth = rboth-rnone
        res['single_gating'] = (dron-droff)/(dron+droff)
        res['multi_gating'] = (dron-droff)/(2*drboth-dron-droff)

        return res

    def plot_vary_n_disinh(self,savenew=False,plot_ylabel=True):
        p = MCM.read_params(self.paramsfile)
        p['inh_rate_base'] = 5
        p['inh_rate_max'] = 30 + p['inh_rate_base']
        p['inh_base'] = p['inh_rate_base']*p['tauGABA']*p['gGABA']/nsiemens  # rate * tau_GABA * gGABA
        p['inh_max'] = p['inh_rate_max']*p['tauGABA']*p['gGABA']/nsiemens  # rate * tau_GABA * gGABA
        p['num_path'] = 2
        p['v_target'] = -40


        single_gating_list = list()

        single_gating_nonoverlap_list = list()
        num_DendDisinh_list = list()
        leg_list = list()


        num_dend_list = [10,20,30]
        for num_dend in num_dend_list:
            num_DendDisinh_plot = arange(3,num_dend)

            single_gating_plot = list()

            for num_DendDisinh in num_DendDisinh_plot:
                p['num_DendEach'] = num_dend
                p['num_DendDisinh'] = num_DendDisinh

                res = self.analytic_evalgating(p)

                single_gating_plot.append(res['single_gating'])


            num_DendDisinh_list.append(num_DendDisinh_plot)
            single_gating_list.append(single_gating_plot)
            leg_list.append(str(num_dend))

            p['num_DendDisinh'] = 3
            res = self.analytic_evalgating(p,no_overlap=True)
            single_gating_nonoverlap_list.append(res['single_gating'])

        # Single path
        fig=MyFigure(figsize=(1.5,1.5))
        pl = fig.addplot(rect=[0.3,0.3,0.65,0.65])
        for i in xrange(len(num_dend_list)):
            pl.plot(num_DendDisinh_list[i]/num_dend_list[i],single_gating_list[i],color=colorMatrix2[i])
        ylim((0,1))
        xlim((0,1))

        xticks([0,0.5,1],['0','0.5','1'])
        xlabel('$N_{disinh}/N_{dend}$')
        if plot_ylabel:
            ylabel('Gating selectivity')
            yticks([0,0.5,1],['0','0.5','1'])
        else:
            yticks([0,0.5,1],['','',''])
        leg = legend(leg_list,title='$N_{dend}$',loc=1, bbox_to_anchor=[1.1, 1.1])
        leg.draw_frame(False)
        for i in xrange(len(num_dend_list)):
            pl.plot([0],single_gating_nonoverlap_list[i],'D',markerfacecolor=colorMatrix2[i],
                    markeredgecolor = 'none',markersize=4)
        if savenew:
            fig.save(self.figpath+'GatingCapacitySinglePathVaryNum'+self.version)

    def plot_vary_inh(self,savenew=True,plot_ylabel=True):
        p = MCM.read_params(self.paramsfile)
        p['inh_rate_base'] = 5


        p['num_soma'] = 1
        p['num_DendEach'] = 10
        p['num_path'] = 2
        p['num_DendDisinh'] = 3

        p['mode'] = 'control_voltage'
        p['v_target'] = -40


        single_gating_list = list()

        single_gating_nonoverlap_list = list()
        num_dend_list = list()
        leg_list = list()
        inh_list = [30,20,10]

        for inh in inh_list:
            p['inh_rate_max'] = inh + p['inh_rate_base']
            p['inh_base'] = p['inh_rate_base']*p['tauGABA']*p['gGABA']/nsiemens  # rate * tau_GABA * gGABA
            p['inh_max'] = p['inh_rate_max']*p['tauGABA']*p['gGABA']/nsiemens  # rate * tau_GABA * gGABA
            num_dend_plot = range(p['num_DendDisinh'],50)

            single_gating_plot = list()

            for num_dend in num_dend_plot:
                p['num_DendEach'] = num_dend

                res = self.analytic_evalgating(p)

                single_gating_plot.append(res['single_gating'])

            num_dend_list.append(num_dend_plot)
            single_gating_list.append(single_gating_plot)
            leg_list.append(str(inh))

            res = self.analytic_evalgating(p,no_overlap=True)
            single_gating_nonoverlap_list.append(res['single_gating'])

        # Single path
        fig=MyFigure(figsize=(1.5,1.5))
        pl = fig.addplot(rect=[0.3,0.3,0.65,0.65])
        for i in xrange(len(inh_list)):
            pl.plot(num_dend_list[i][0]/array(num_dend_list[i]),single_gating_list[i],
                    color=colorMatrix1[2*i+1])
        ylim((0,1))
        xlim((0,1))
        xticks([0,0.5,1],['0','0.5','1'])
        xlabel('$N_{disinh}/N_{dend}$')
        if plot_ylabel:
            ylabel('Gating selectivity')
            yticks([0,0.5,1],['0','0.5','1'])
        else:
            yticks([0,0.5,1],['','',''])
        leg = legend(leg_list,title='Disinhibition (Hz)',loc=1, bbox_to_anchor=[1.1, 1.1])
        leg.draw_frame(False)
        for i in xrange(len(inh_list)):
            pl.plot([0],single_gating_nonoverlap_list[i],'D',markerfacecolor=colorMatrix1[2*i+1],
                    markeredgecolor = 'none',markersize=4)
        if savenew:
            fig.save(self.figpath+'GatingCapacitySinglePathVaryInh'+self.version)

    def run_NMDAAMPAratio(self,inh_rate=30*Hz):
        p = dict()
        p['pre_rates'] = [array([40,40,0,0]),array([40,0,40,0])]

        p['w_input'] = (1,0)
        p['num_input'] = 15
        p['bkg_inh_rate'] = 5*Hz
        p['dend_inh_rate'] = inh_rate

        num_soma = len(p['pre_rates'][0])
        p['num_soma'] = num_soma
        p['runtime'] = 500*second

        p['dt'] = 0.2*ms
        p['record_dt'] = 1.0*ms
        p['num_DendEach'] = 10

        NMDA_prop_plot = linspace(0,1,6)
        gs1_plot = list()
        gs2_plot = list()
        firing_rate_list = list()
        for NMDA_prop in NMDA_prop_plot:
            model = MCM.Model(paramsfile=self.paramsfile, eqsfile=self.eqsfile,outsidePara=p)
            model.make_model(num_soma = num_soma, condition = 'invivo')
            model.make_invivo_bkginput()
            model.make_dend_bkginh(inh_rates=p['bkg_inh_rate'])
            model.two_pathway_gating_experiment(p['pre_rates'], num_input=p['num_input'],
                                                dend_inh_rate=p['dend_inh_rate'],
                                                w_input = p['w_input'], NMDA_prop = NMDA_prop)
            model.make_network()
            model.reinit()
            net = Network(model)
            net.run(p['runtime'],report='text')
            mon = model.monitor

            firing_rate_plot = zeros(num_soma)
            for i in xrange(num_soma):
                firing_rate_plot[i] = sum(mon['MSpike'][i]>0.5)/(p['runtime']-0.5)

            fr = firing_rate_plot

            # gating selectivity
            # In single pathway condition
            gs1 = ((fr[1]-fr[3])-(fr[2]-fr[3]))/((fr[1]-fr[3])+(fr[2]-fr[3]))
            gs2 = ((fr[0]-fr[2])-(fr[0]-fr[1]))/((fr[0]-fr[2])+(fr[0]-fr[1]))
            gs1_plot.append(gs1)
            gs2_plot.append(gs2)
            firing_rate_list.append(firing_rate_plot)


        p['NMDA_prop_plot'] = NMDA_prop_plot
        p['gs1_plot'] = gs1_plot
        p['gs2_plot'] = gs2_plot
        p['firing_rate_list'] = firing_rate_list

        with open(self.datapath+'GatingCapacity_varyNMDAAMPAratio'+self.version,'wb') as f:
            pickle.dump(p,f)

    def plot_NMDAAMPAratio(self):
        with open(self.datapath+'GatingCapacity_varyNMDAAMPAratio'+self.version,'rb') as f:
            p = pickle.load(f)


        fig=MyFigure(figsize=(1.5,1.5))
        pl = fig.addplot(rect=[0.3,0.3,0.65,0.65])
        pl.plot(1-p['NMDA_prop_plot'],p['gs1_plot'],
        'o-', color=colorMatrix1[1],
        markeredgecolor = 'none', markersize=3)
        ylabel('Gating selectivity')
        ylim((0,1))
        xlim((0,1))
        yticks([0,0.5,1],['0','0.5','1'])
        xticks([0,0.5,1],['0','0.5','1'])
        xlabel('AMPA conductance ratio')
        fig.save(self.figpath+'GatingCapacity_varyNMDAAMPAratio'+self.version)
        '''
        p_list = list()
        for filename in filenames:
            with open(self.datapath+''+filename,'rb') as f:
                p = pickle.load(f)
                p_list.append(p)

        storename = ['Single','Multi']
        for i_type in [1,2]:
            fig=MyFigure(figsize=(1.5,1.5))
            pl = fig.addplot(rect=[0.3,0.3,0.65,0.65])
            i=0
            for p in p_list:
                pl.plot(1-p['NMDA_prop_plot'],p['gs%d_plot' % i_type],
                'o-',label='%d' % p['dend_inh_rate'], color=colorMatrix1[2*i+1],
                markeredgecolor = 'none', markersize=3)
                ylabel('Gating selectivity')
                ylim((0,1))
                xlim((0,1))
                yticks([0,0.5,1],['0','0.5','1'])
                xticks([0,0.5,1],['0','0.5','1'])
                xlabel('AMPA conductance ratio')
                leg = legend(title='Disinhibition (Hz)',loc=1, bbox_to_anchor=[1.1, 1.1])
                leg.draw_frame(False)
                i+=1
            fig.save(self.figpath+'GatingCapacity'+storename[i_type-1]+'VaryAMPA_'+self.version)
        '''

    def run_GABAABratio(self,inh_rate=30*Hz):
        p = dict()
        n_repeat = 30
        p['pre_rates'] = [array([40,0,0]).repeat(n_repeat),
                          array([0,40,0]).repeat(n_repeat)]

        p['w_input'] = (1,0)
        p['num_input'] = 15
        p['bkg_inh_rate'] = 5*Hz
        p['dend_inh_rate'] = inh_rate

        num_soma = len(p['pre_rates'][0])
        p['num_soma'] = num_soma
        p['runtime'] = 20*second

        p['dt'] = 0.2*ms
        p['record_dt'] = 1.0*ms
        p['num_DendEach'] = 10

        GABAA_prop_plot = linspace(0,1,6)
        gs1_plot = list()
        firing_rate_list = list()
        for GABAA_prop in GABAA_prop_plot:
            model = MCM.Model(paramsfile=self.paramsfile, eqsfile=self.eqsfile,
                              outsidePara=p)
            model.make_model(num_soma = num_soma, condition = 'invivo',record_all=False)
            model.make_invivo_bkginput()
            model.make_dend_bkginh(inh_rates=p['bkg_inh_rate'])
            model.two_pathway_gating_experiment(p['pre_rates'], num_input=p['num_input'],
                                                dend_inh_rate=p['dend_inh_rate'],
                                                w_input = p['w_input'], GABAA_prop = GABAA_prop)
            model.make_network()
            model.reinit()
            net = Network(model)
            net.run(p['runtime'],report='text')
            mon = model.monitor

            firing_rate_plot = zeros(num_soma)
            for i in xrange(num_soma):
                firing_rate_plot[i] = sum(mon['MSpike'][i]>0.5)/(p['runtime']-0.5)

            fr = firing_rate_plot.reshape((num_soma//n_repeat,n_repeat)).mean(axis=1)
            fr = abs(fr-fr[-1])
            # gating selectivity
            # In single pathway condition
            gs1 = (fr[0]-fr[1])/(fr[0]+fr[1]+1e-5)
            gs1_plot.append(gs1)
            firing_rate_list.append(firing_rate_plot)


        p['GABAA_prop_plot'] = GABAA_prop_plot
        p['gs1_plot'] = gs1_plot
        p['firing_rate_list'] = firing_rate_list

        with open(self.datapath+'GatingCapacity_varyGABAABratio'+self.version,'wb') as f:
            pickle.dump(p,f)

    def plot_GABAABratio(self):
        with open(self.datapath+'GatingCapacity_varyGABAABratio'+self.version,'rb') as f:
            p = pickle.load(f)


        fig=MyFigure(figsize=(1.5,1.5))
        pl = fig.addplot(rect=[0.3,0.3,0.65,0.65])
        pl.plot(1-p['GABAA_prop_plot'],p['gs1_plot'],
        'o-', color=colorMatrix1[1],
        markeredgecolor = 'none', markersize=3)
        print p['gs1_plot']
        ylabel('Gating selectivity')
        ylim((0,1))
        xlim((0,1))
        yticks([0,0.5,1],['0','0.5','1'])
        xticks([0,0.5,1],['0','0.5','1'])
        xlabel(r'$\mathrm{GABA}_{\mathrm{B}}$ conductance ratio')
        fig.save(self.figpath+'GatingCapacity_varyGABAABratio'+self.version)

    def run_NMDAAMPAGABAABratio(self,inh_rate=30*Hz):
        p = dict()
        n_repeat = 30
        p['pre_rates'] = [array([40,0,0]).repeat(n_repeat),
                          array([0,40,0]).repeat(n_repeat)]

        p['w_input'] = (1,0)
        p['num_input'] = 15
        p['bkg_inh_rate'] = 5*Hz
        p['dend_inh_rate'] = inh_rate

        num_soma = len(p['pre_rates'][0])
        p['num_soma'] = num_soma
        p['runtime'] = 20*second

        p['dt'] = 0.5*ms
        p['record_dt'] = 1.0*ms
        p['num_DendEach'] = 10

        GABAA_prop_plot = linspace(0,1,6)
        NMDA_prop_plot = linspace(0,1,6)
        GABAA_prop_plot_exp = GABAA_prop_plot.repeat(len(NMDA_prop_plot))
        NMDA_prop_plot_exp = np.tile(NMDA_prop_plot,len(GABAA_prop_plot))
        
        gs1_plot = list()
        firing_rate_list = list()
        for GABAA_prop, NMDA_prop in zip(GABAA_prop_plot_exp,NMDA_prop_plot_exp):
            model = MCM.Model(paramsfile=self.paramsfile, eqsfile=self.eqsfile,
                              outsidePara=p)
            model.make_model(num_soma = num_soma, condition = 'invivo',record_all=False)
            model.make_invivo_bkginput()
            model.make_dend_bkginh(inh_rates=p['bkg_inh_rate'])
            model.two_pathway_gating_experiment(p['pre_rates'], num_input=p['num_input'],
                                                dend_inh_rate=p['dend_inh_rate'],
                                                w_input = p['w_input'],
                                                NMDA_prop = NMDA_prop,GABAA_prop = GABAA_prop)
            model.make_network()
            model.reinit()
            net = Network(model)
            net.run(p['runtime'],report='text')
            mon = model.monitor

            firing_rate_plot = zeros(num_soma)
            for i in xrange(num_soma):
                firing_rate_plot[i] = sum(mon['MSpike'][i]>0.5)/(p['runtime']-0.5)

            fr = firing_rate_plot.reshape((num_soma//n_repeat,n_repeat)).mean(axis=1)
            fr = abs(fr-fr[-1])
            # gating selectivity
            # In single pathway condition
            gs1 = (fr[0]-fr[1])/(fr[0]+fr[1]+1e-5)
            gs1_plot.append(gs1)
            firing_rate_list.append(firing_rate_plot)


        p['GABAA_prop_plot'] = GABAA_prop_plot
        p['NMDA_prop_plot'] = NMDA_prop_plot
        p['GABAA_prop_plot_exp'] = GABAA_prop_plot_exp
        p['NMDA_prop_plot_exp'] = NMDA_prop_plot_exp
        p['gs1_plot'] = gs1_plot
        p['firing_rate_list'] = firing_rate_list
        p['n_repeat'] = n_repeat

        with open(self.datapath+'GatingCapacity_varyNMDAAMPAGABAABratio'+self.version,'wb') as f:
            pickle.dump(p,f)

    def plot_NMDAAMPAGABAABratio(self):
        with open(self.datapath+'GatingCapacity_varyNMDAAMPAGABAABratio'+self.version,'rb') as f:
            p = pickle.load(f)

        gs_plot = p['gs1_plot'][::-1]
        GABAA_prop_plot = p['GABAA_prop_plot']
        NMDA_prop_plot = p['NMDA_prop_plot']
        gs_plot = np.reshape(gs_plot,(len(GABAA_prop_plot),len(NMDA_prop_plot)))

        fig = plt.figure(1,(2.4,2.4))
        ax = fig.add_axes([0.2,0.3,0.5,0.5])
        im0 = ax.imshow(gs_plot, interpolation='none',cmap='hot',extent=(0,1,0,1),vmin=0,vmax=1,origin='lower')
        ax.get_xaxis().set_ticks([0,0.5,1])
        ax.get_yaxis().set_ticks([0,0.5,1])
        ax.set_xlabel('AMPA conductance ratio')
        ax.set_ylabel(r'$\mathrm{GABA}_{\mathrm{B}}$ conductance ratio')
        #cbar = plt.colorbar(im0, format="%d", cax = axes([0.75, 0.2, 0.05, 0.6]),ticks=(minr,(maxr//5)*5))
        cbar = plt.colorbar(im0, format="%0.1f", cax = axes([0.75, 0.3, 0.05, 0.5]),ticks=(0,0.5,1))
        cbar.set_label('Gating selectivity',rotation=270,fontsize=7,labelpad=10)
        savefig(self.figpath+'GatingCapacity_varyNMDAAMPAGABAABratio'+self.version+'.pdf',format='pdf')