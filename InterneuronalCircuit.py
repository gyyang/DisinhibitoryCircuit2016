# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:44:28 2015

@author: guangyuyang

Here we study the plausibility of the disinhibitory circuit

When varying proportion of cells targeted, normalize by strength
"""
from __future__ import division
import numpy as np
import scipy as sp
import scipy.optimize
import copy
import time
import pickle
import pylab
import datetime
import matplotlib.pyplot as plt 
import MultiCompartmentalModel as MCM
from figtools import MyFigure, MySubplot

dend_IO = MCM.dend_IO
soma_IO = MCM.soma_IO
soma_fv = MCM.soma_fv
soma_fI = MCM.soma_fI

plot = pylab.plot

class Study():
    def __init__(self,figpath='figure/', datapath='data/', version=None,
                 paramsfile='parameters.txt',eqsfile='equations.txt'):
        self.figpath = figpath
        self.datapath = datapath
        if version is None:
            version = '_'+str(datetime.date.today())

        self.version = version

        self.paramsfile = paramsfile
        self.eqsfile = eqsfile

        self.params = MCM.read_params(paramsfile)

        self.gCouple = self.params['num_DendEach']*self.params['gEachCouple_pyr_vivo']*1e9


    def plot_inputs(self,savenew=True,modeltype='Control2VIPSOM'):

        p = self.get_p(modeltype, outside_params=dict())
        result = self.get_gatingselectivity(p)

        n_som_show = 20

        figsize = (1.3,1.2)
        pl_top =  [0.25,0.6,0.7,0.35]
        pl_bottom =  [0.25,0.25,0.7,0.35]

        fig = MyFigure(figsize=figsize)
        pl=fig.addplot(pl_top)
        pl.bar(np.arange(n_som_show),result['Exc2som_list'][0][:n_som_show],
               linewidth=0,color=np.array([228,26,28])/255,edgecolor=np.array([228,26,28])/255)
        pl.ax.set_xticks([])
        pl.ax.set_yticks([0,200])

        pl=fig.addplot(pl_bottom)
        pl.bar(np.arange(n_som_show),result['Inh2som_list'][0][:n_som_show],
               linewidth=0,color=np.array([55,126,184])/255,edgecolor=np.array([55,126,184])/255)
        pl.ax.set_xticks([0,5,10,15,20])
        pl.ax.set_yticks([0,200])
        pl.ax.set_yticklabels(['0','-200'])
        pl.ax.invert_yaxis()
        pl.xlabel('SOM Neurons')
        #pl.ax.spines['bottom'].set_visible(False)
        if savenew:
            fig.save(self.figpath+'GatingSelectivity_EIInput2som'+'_'+modeltype+self.version)

        input2som = result['Exc2som_list'][0][:n_som_show]-result['Inh2som_list'][0][:n_som_show]
        fig = MyFigure(figsize=figsize)
        pl=fig.addplot(pl_top)
        pl.bar(np.arange(n_som_show),input2som*(input2som>0),
               linewidth=0,color=np.array([152,78,163])/255,edgecolor=np.array([152,78,163])/255)
        pl.ax.set_xticks([])
        pl.ax.set_yticks([0,200])
        pl.ax.set_yticklabels([''])
        pl=fig.addplot(pl_bottom)
        pl.bar(np.arange(n_som_show),-input2som*(input2som<0),
               linewidth=0,color=np.array([152,78,163])/255,edgecolor=np.array([152,78,163])/255)
        pl.ax.set_xticks([0,5,10,15,20])
        pl.ax.set_yticks([0,200])
        pl.ax.set_yticklabels(['',''])
        pl.ax.invert_yaxis()
        pl.xlabel('SOM Neurons')
        #pl.ax.spines['bottom'].set_visible(False)
        if savenew:
            fig.save(self.figpath+'GatingSelectivity_TotalInput2som'+'_'+modeltype+self.version)


        n_dend_show = 20
        fig = MyFigure(figsize=figsize)
        pl=fig.addplot(pl_top)
        pl.bar(np.arange(n_dend_show),result['Exc2dend_list'][0][:n_dend_show],
               linewidth=0,color=np.array([228,26,28])/255,edgecolor=np.array([228,26,28])/255)
        pl.ax.set_xticks([])
        pl.ax.set_yticks([0,30])

        pl=fig.addplot(pl_bottom)
        pl.bar(np.arange(n_dend_show),result['Inh2dend_list'][0][:n_dend_show],
               linewidth=0,color=np.array([55,126,184])/255,edgecolor=np.array([55,126,184])/255)
        pl.ax.set_xticks([0,5,10,15,20])
        pl.ax.set_ylim((0,4))
        pl.ax.set_yticks([4])
        pl.ax.set_yticklabels(['4'])
        pl.ax.invert_yaxis()
        pl.xlabel('Dendrites')
        #pl.ax.spines['bottom'].set_visible(False)

        if savenew:
            fig.save(self.figpath+'GatingSelectivity_EIInput2dend'+'_'+modeltype+self.version)

        fig = MyFigure(figsize=figsize)
        pl=fig.addplot(pl_top)
        pl.bar(np.arange(n_dend_show),result['Exc2dend_list'][1][:n_dend_show],
               linewidth=0,color=np.array([228,26,28])/255,edgecolor=np.array([228,26,28])/255)
        pl.ax.set_xticks([])
        pl.ax.set_yticks([0,30])

        pl=fig.addplot(pl_bottom)
        pl.bar(np.arange(n_dend_show),result['Inh2dend_list'][0][:n_dend_show],
               linewidth=0,color=np.array([55,126,184])/255,edgecolor=np.array([55,126,184])/255)
        pl.ax.set_xticks([0,5,10,15,20])
        pl.ax.set_ylim((0,4))
        pl.ax.set_yticks([4])
        pl.ax.set_yticklabels(['4'])
        pl.ax.invert_yaxis()
        pl.xlabel('Dendrites')
        #pl.ax.spines['bottom'].set_visible(False)

        if savenew:
            fig.save(self.figpath+'GatingSelectivity_EIInput2dend_offpath'+'_'+modeltype+self.version)

        outside_param = {'with_PV':True}
        p = self.get_p(modeltype=modeltype,
                     outside_params=outside_param)
        w_som2pv = 0.1
        w_som2pv_total = w_som2pv*p['n_som']*p['p_som2pv']
        p['w_som2pv_total'] = w_som2pv_total
        result = self.get_gatingselectivity(p)

        n_pv_show = 20
        input_som2pv = -result['Inh_som2pv_list'][0][:n_pv_show]
        fig = MyFigure(figsize=figsize)
        pl=fig.addplot(pl_top)
        pl.bar(np.arange(n_pv_show),input_som2pv*(input_som2pv>0),
               linewidth=0,color=np.array([152,78,163])/255,edgecolor=np.array([152,78,163])/255)
        pl.ax.set_xticks([])
        pl.ax.set_yticks([0,100])
        pl.ax.set_yticklabels(['','100'])
        pl=fig.addplot(pl_bottom)
        pl.bar(np.arange(n_pv_show),-input_som2pv*(input_som2pv<0),
               linewidth=0,color=np.array([152,78,163])/255,edgecolor=np.array([152,78,163])/255)
        pl.ax.set_xticks([0,5,10,15,20])
        pl.ax.set_yticks([0,100])
        pl.ax.set_yticklabels(['0','-100'])
        pl.ax.invert_yaxis()
        pl.xlabel('PV Neurons')
        #pl.ax.spines['bottom'].set_visible(False)
        if savenew:
            fig.save(self.figpath+'GatingSelectivity_TotalInh_som2pv'+'_'+modeltype+self.version)

        n_soma_show = 20
        input2soma = -result['Inh2soma_list'][0][:n_soma_show]
        fig = MyFigure(figsize=figsize)
        pl=fig.addplot(pl_top)
        pl.bar(np.arange(n_som_show),input2soma*(input2soma>0),
               linewidth=0,color=np.array([152,78,163])/255,edgecolor=np.array([152,78,163])/255)
        pl.ax.set_xticks([])
        pl.ax.set_yticks([0,100])
        pl.ax.set_yticklabels(['','100'])
        pl=fig.addplot(pl_bottom)
        pl.bar(np.arange(n_som_show),-input2soma*(input2soma<0),
               linewidth=0,color=np.array([152,78,163])/255,edgecolor=np.array([152,78,163])/255)
        pl.ax.set_xticks([0,5,10,15,20])
        pl.ax.set_yticks([0,100])
        pl.ax.set_yticklabels(['0','-100'])
        pl.ax.invert_yaxis()
        pl.xlabel('Somas')
        #pl.ax.spines['bottom'].set_visible(False)
        if savenew:
            fig.save(self.figpath+'GatingSelectivity_TotalInput2soma'+'_'+modeltype+self.version)


    def get_exc(self,p,inh):
        '''
        W is SOM to dend connections with units nS
        r_som in the activation pattern in Hz
        return arrays of exc and inh in units nS
        '''
        if p['exc_type'] == 'binary': # binary
            exc = (inh<p['inh_threshold'])*p['g_exc']
            # if inh is weaker than inh_threshold, then exc is g_exc, else 0
        elif p['exc_type'] == 'invprop_inh':
            # excitatory pattern inversely proportional to inhibition pattern
            exc = (p['inh_threshold']-inh)*(p['inh_threshold']>inh)/p['inh_threshold']*p['g_exc']
        elif p['exc_type'] == 'sigmoid':
            exc = p['g_exc']/(1+np.exp((inh-p['inh_threshold'])/p['inh_spread']))
        return exc

    def generate_W(self,n_from,n_to,conn_density,syn_weight_total,method='strict'):
        '''
        generate connectivity matrix n_to * n_from
        syn_weight_total: total synaptic weight onto one postsynaptic neuorn
        '''

        # Number of SOM synapses onto each dendrite
        n_from2to = conn_density*n_from
        if method == 'strict':
            syn_weight = syn_weight_total/round(n_from2to)
        else:
            syn_weight = syn_weight_total/n_from2to

        if method=='strict':
            W = np.zeros((n_to,n_from))
            W[:,:round(conn_density*n_from)] = syn_weight # conductance of each synapse
            map(np.random.shuffle, W)
        elif method=='strict_interp':
            W = np.zeros((n_to,n_from))
            n_from_conn = np.floor(conn_density*n_from)
            W[:,:n_from_conn] = syn_weight # conductance of each synapse
            if n_from_conn < n_from:
                W[:,n_from_conn] = syn_weight*(conn_density*n_from-n_from_conn) # conductance of each synapse
            map(np.random.shuffle, W)
        elif method == 'old_random':
            W = np.zeros((n_to,n_from))
            K = np.random.rand(n_to,n_from)
            W[K<conn_density] = syn_weight
        elif method == 'random':
            W = np.zeros(n_to*n_from)
            W[:round(conn_density*n_to*n_from)] = syn_weight
            np.random.shuffle(W)
            W = np.reshape(W,(n_to,n_from))
        elif method == 'temp':
            Mask = np.zeros((n_to,n_from))
            Mask[:,:round(conn_density*n_from)] = 1 # conductance of each synapse
            map(np.random.shuffle, Mask)
            W = Mask * np.random.rand(n_to,n_from)*2*syn_weight
        return W

    def generate_W_specific_dend(self,p):
        # Each SOM cell target a specific dendrite
        # SOM cell close to each other in terms of index, will target similar dendrites
        n_pyr_conn = round(p['p_som2pyr']*p['n_pyr']) # here conn_density is cell-cell level
        ind_pyr = np.arange(p['n_pyr'])
        W = np.zeros((p['n_pyr']*p['n_dend_each'],p['n_som']))
        p['g_inh'] = p['g_total']/(p['p_som2pyr']*p['n_som']/p['n_dend_each'])

        for i_som in np.arange(p['n_som']): # Current SOM cell
            np.random.shuffle(ind_pyr)
            ind_pyr_conn = ind_pyr[:n_pyr_conn] # index of pyramidal cells targeted
            # Here assume each SOM cell only target two dendrites on each pyramidal cell
            # Index of dendrite targeted by this SOM cell
            ind_dend = (i_som/p['n_som'])*p['n_dend_each']
            ind_dend_1 = np.int(np.floor(ind_dend)) # smaller index of the two targetd SOM cell
            W[ind_pyr_conn*p['n_dend_each']+ind_dend_1,i_som] = p['g_inh']*(ind_dend_1+1-ind_dend)
            if ind_dend_1 < p['n_dend_each']-1:
                W[ind_pyr_conn*p['n_dend_each']+ind_dend_1+1,i_som] = p['g_inh']*(ind_dend-ind_dend_1)

        return W

    def fI_SOM(self,current): # current with units pA
        rheobase = 40 #pA, neuroelectro.org
        #fIslope = 36./1000 # Hz/nA, neuroelectro.org
        fIslope = 90./1000 # Hz/nA, neuroelectro.org
        rate = (current-rheobase)*fIslope
        rate = rate*(rate>0) # Hz
        return rate

    def get_r_som(self,p):
        Exc2som_list = list()
        Inh2som_list = list()
        r_som_list = list()

        n_exc2som = round(p['p_exc2som']*p['n_som'])
        n_exc2vip = round(p['p_exc2vip']*p['n_vip'])

        n_vip2som = p['p_vip2som']*p['n_vip']
        # uIPSQ is about 0.7 pC=0.7 pA/Hz for VIP-SOM connection, Pfeffer et al. Nat Neurosci. 2012
        #syn_weight_vip2som = 10/n_vip2som
        p['w_vip2som'] = p['w_vip2som_total']/n_vip2som
        #p['w_vip2som'] = 0.7
        #syn_weight_vip2som = 0.7
        W_vip2som = self.generate_W(p['n_vip'],p['n_som'],p['p_vip2som'],
                                   p['w_vip2som_total'],method='strict_interp')

        for i_path in range(p['n_path']):
            Exc2som = np.zeros(p['n_som'])
            Exc2som[:n_exc2som] = p['exc_som_mean']*p['n_som']/n_exc2som
            np.random.shuffle(Exc2som)

            r_vip = np.zeros(p['n_vip'])
            #r_vip[:n_exc2vip] = p['r_vip_max']
            r_vip[:n_exc2vip] = p['r_vip_mean']*p['n_vip']/n_exc2vip
            np.random.shuffle(r_vip)
            Inh2som = np.dot(W_vip2som,r_vip)
            input2som = Exc2som - Inh2som + p['Exc2som0']

            r_som_list.append(self.fI_SOM(input2som))
            Exc2som_list.append(Exc2som)
            Inh2som_list.append(Inh2som)

        return r_som_list, Exc2som_list, Inh2som_list

    def get_r_som_novip(self,p):
        Exc2som_list = list()
        Inh2som_list = list()
        r_som_list = list()

        n_exc2som = round(p['p_exc2som']*p['n_som'])

        for i_path in range(p['n_path']):
            Inh2som = np.ones(p['n_som'])*p['Exc2som0']
            Exc2som = np.zeros(p['n_som'])
            Exc2som[:n_exc2som] = p['Exc2som0']
            np.random.shuffle(Exc2som)
            input2som = Exc2som - Inh2som + p['Exc2som0']

            r_som_list.append(self.fI_SOM(input2som))
            Exc2som_list.append(Exc2som)
            Inh2som_list.append(Inh2som)

        return r_som_list, Exc2som_list, Inh2som_list

    def Get_r_fromV(self,Exc,Inh,n_dend_each):
        DendV = dend_IO(Exc, Inh)
        MeanDendV = DendV.reshape(len(DendV)//n_dend_each,n_dend_each).mean(axis=1)
        SomaR = soma_fv(MeanDendV)
        return SomaR

    def Get_r(self,Exc,Inh,Inh2soma,n_dend_each):
        # Get rate from injection current
        DendV = dend_IO(Exc, Inh)
        MeanDendV = DendV.reshape(len(DendV)//n_dend_each,n_dend_each).mean(axis=1)
        vSoma = -55 # Assume somatic voltage is around the reset, which is a good approximation
        SomaR = soma_fI(self.gCouple*(MeanDendV-vSoma)-Inh2soma)
        return SomaR

    def get_inh2soma(self,r_som,p):
        '''
        Get inhibitory current to soma given the activation of
        :param r_som: the activity of SOM neurons
        :param p:
        :return:
        '''
        Inh_pv2pyr = p['I_PV'] # directly set the somatic inhibition
        Inh_som2pv = 0
        dr_pv = 0

        if p['with_PV']: # calculate the somatic inhibition through PV
            # Notice PV activity increase, so we can disregard the rectification
            fIslope_pv = 220./1000 # Hz/nA, neuroelectro.org
            r_som0 = self.fI_SOM(p['Exc2som0'])
            dr_som = r_som-r_som0 # These numbers should be primarily negative
            Inh_som2pv = np.dot(p['W_som2pv'],dr_som)
            temp_mat = np.eye(p['n_pv'])/fIslope_pv+p['W_pv2pv'] # Notice the plus sign because PV is inhibitory
            dr_pv = np.linalg.solve(temp_mat,-Inh_som2pv) # These numbers should be positive
            Inh_pv2pyr += np.dot(p['W_pv2pyr'],dr_pv)
            if p['I_PV']>0:
                print('WARNING: Direct I_PV not zero, though PV circuit is used')

        return Inh_pv2pyr, Inh_som2pv, dr_pv

    def get_gatingselectivity(self,p,verbose=False):
        '''
        disinh_prop:  Proportion of disinhibition compare to the inhibition level of baseline
        '''

        result = dict()

        # Derived parameters
        p['n_dend'] = p['n_dend_each']*p['n_pyr']
        p['p_som2dend'] = 1-(1-p['p_som2pyr'])**(1./p['n_dend_each'])
        # Number of SOM synapses onto each dendrite
        p['n_som2dend'] = p['p_som2dend']*p['n_som']

        start = time.time()
        if p['gen_W_method'] is 'specific_dend':
            W_som2dend = self.generate_W_specific_dend(p)
        else:
            W_som2dend = self.generate_W(p['n_som'],p['n_dend'],p['p_som2dend'],p['g_total'],method=p['gen_W_method'])

        # Activity pattern of SOM cells
        if not p['with_VIP']:
            r_som_list, Exc2som_list, Inh2som_list = self.get_r_som_novip(p)
        elif ('grid_on' in p) and p['grid_on']:
            r_som_list, Exc2som_list, Inh2som_list = self.get_r_som_grid(p)
        else:
            r_som_list, Exc2som_list, Inh2som_list = self.get_r_som(p)

        # PV neurons
        if p['with_PV']:
            p['W_som2pv'] = self.generate_W(p['n_som'],p['n_pv'],p['p_som2pv'],p['w_som2pv_total'],method=p['gen_W_method'])
            p['W_pv2pv']  = self.generate_W(p['n_pv'], p['n_pv'],p['p_pv2pv'], p['w_pv2pv_total'], method=p['gen_W_method'])
            p['W_pv2pyr'] = self.generate_W(p['n_pv'],p['n_pyr'],p['p_pv2pyr'],p['w_pv2pyr_total'],method=p['gen_W_method'])


        Exc2dend_list = list()
        Inh2dend_list = list()
        Exc2dend_total = 0

        Inh2soma_list = list()
        Inh_som2pv_list = list()
        dr_pv_list = list()

        for i_path in range(p['n_path']):
            Inh2dend = np.dot(W_som2dend,r_som_list[i_path])*self.params['tauGABA'] # nS
            Exc2dend = self.get_exc(p,Inh2dend) # nS
            Exc2dend_list.append(Exc2dend)
            Inh2dend_list.append(Inh2dend)
            Exc2dend_total += Exc2dend

            Inh2soma, Inh_som2pv, dr_pv = self.get_inh2soma(r_som_list[i_path],p) # pA
            Inh2soma_list.append(Inh2soma)
            Inh_som2pv_list.append(Inh_som2pv)
            dr_pv_list.append(dr_pv)



        get_r = lambda exc,inh,inh2soma : self.Get_r(exc,inh,inh2soma,p['n_dend_each'])
        Exc0 = Exc2dend_list[0]*0
        ron, roff, rnone, rboth = 0,0,0,0
        for i_path in range(p['n_path']):
            ron += get_r(Exc2dend_list[i_path],Inh2dend_list[i_path],Inh2soma_list[i_path])/p['n_path'] # Gated-on pathway on
            roff += get_r(Exc2dend_total-Exc2dend_list[i_path],Inh2dend_list[i_path],Inh2soma_list[i_path])/p['n_path'] # Every other pathways on
            rnone += get_r(Exc0,Inh2dend_list[i_path],Inh2soma_list[i_path])/p['n_path']
            rboth += get_r(Exc2dend_total,Inh2dend_list[i_path],Inh2soma_list[i_path])/p['n_path']

        dron = ron-rnone
        droff = roff-rnone
        eps = 1e-5
        gs_list = np.array((dron-droff)/(dron+droff+eps))
        # with PV, dron and droff may be of opposite sign, here check it
        gs_list = (gs_list>0)*(gs_list<1)*gs_list+(gs_list>1)*1.
        if gs_list.max()>1 or gs_list.min()<0:
            print('Erroneous gating selectivity. Max {:0.4f}, Min {:0.4f}').format(gs_list.max(),gs_list.min())

        gs1 = gs_list.mean()
        gs2 = (dron.mean()-droff.mean())/(dron.mean()+droff.mean()+eps)


        if p['gating_selectivity_type'] == 'gs_of_mean':
            gs = gs2
        elif p['gating_selectivity_type'] == 'mean_of_gs':
            gs = gs1

        result['param'] = p
        for temp_list, name in zip([gs,gs_list,ron,roff,rnone,rboth,W_som2dend,
                                    Exc2dend_list,Inh2dend_list,r_som_list,
                                    Exc2som_list,Inh2som_list,Inh2soma_list,
                                    Inh_som2pv_list, dr_pv_list],
                                   ['gs','gs_list','ron','roff','rnone','rboth','W_som2dend',
                                    'Exc2dend_list','Inh2dend_list','r_som_list',
                                    'Exc2som_list','Inh2som_list','Inh2soma_list',
                                    'Inh_som2pv_list', 'dr_pv_list']):
            result[name] = np.array(temp_list)

        if verbose:
            print 'n_som2dend = %0.2f' % p['n_som2dend']
            print 'time taken %0.7f s' % (time.time()-start)
            print 'Type %d: gs1 is %0.5f, gs2 is %0.5f' % (p['gating_type'],gs1,gs2)

        return result

    def vary_x(self,param,var_name,var_plot):
        p = copy.deepcopy(param)
        gs_plot = list()
        gs_low_plot = list() # Lower quantile
        gs_high_plot = list() # Higher quantile
        for var in var_plot:
            p[var_name] = var
            result = self.get_gatingselectivity(p)
            gs_plot.append(result['gs'])
            gs_list_sorted = np.sort(result['gs_list'])
            gs_low_plot.append(gs_list_sorted[round(0.1*len(gs_list_sorted))])
            gs_high_plot.append(gs_list_sorted[round(0.9*len(gs_list_sorted))])
            print var_name + '=%0.3f, gs=%0.3f' % (var,result['gs'])

        return np.array(gs_plot), np.array(gs_low_plot), np.array(gs_high_plot)

    def vary_x2(self,param,var_name_1,var_plot_1,var_name_2,var_plot_2):
        p = copy.deepcopy(param)
        gs_plot = list()
        gs_low_plot = list() # Lower quantile
        gs_high_plot = list() # Higher quantile
        for var_1, var_2 in zip(var_plot_1,var_plot_2):
            p[var_name_1] = var_1
            p[var_name_2] = var_2
            result = self.get_gatingselectivity(p)
            gs_plot.append(result['gs'])
            gs_list_sorted = np.sort(result['gs_list'])
            gs_low_plot.append(gs_list_sorted[round(0.1*len(gs_list_sorted))])
            gs_high_plot.append(gs_list_sorted[round(0.9*len(gs_list_sorted))])
            print var_name_1 + '=%0.3f' % (var_1),
            print var_name_2 + '=%0.3f' % (var_2),
            print 'gs=%0.3f' % result['gs']

        return np.array(gs_plot), np.array(gs_low_plot), np.array(gs_high_plot)

    def get_p(self, modeltype, outside_params=dict()):
        # Model parameters
        p = dict()
        # Neurons of neurons in 400mum*400mum of L2/3
        p['n_som'] = 160 # 150 SOM neurons is roughly correct
        p['n_vip'] = 140 # Estimate 140 VIP neurons
        p['n_pyr'] = 3000
        p['n_dend_each'] = 30
        #p['n_dend_each'] = 10
        p['p_vip2som'] = 0.6
        p['p_som2pyr'] = 0.6
        p['gen_W_method'] = 'strict_interp' #'random': Avoid non-monotonicity when vary n_dend_each
        p['g_total'] = 40 #nS total inhibitory conductance (nS) received by each dendrite
        p['exc_type'] = 'invprop_inh'
        p['inh_threshold'] = 4.0 #nS
        p['Exc2som0'] = 150 # nA Baseline excitation that keeps SOM firing
        p['g_exc'] = 25 #nS
        p['disinh_prop'] = 0.5
        p['type_som'] = 'random_overlap' # random overlap allows for arbitrary number of pathways
        p['gating_selectivity_type'] = 'mean_of_gs'
        # mean of gs is better because we don't want just a few cells receiving inputs
        p['n_path'] = 2
        p['r_vip_mean'] = 5 # Hz, which activation of VIP neurons


        if modeltype == 'SOM_alone':
            p['with_VIP'] = False
            p['p_exc2som'] = 0.5 # Proportion of SOM targeted by control input for each pathway
        elif modeltype == 'Control2VIPSOM':
            p['exc_som_mean'] = 75 # pA
            p['w_vip2som_total'] = 30 # pA/Hz, total inhibitory weight received by each SOM neuron
            p['p_exc2som'] = 0.5 # Proportion of SOM targeted by control input for each pathway
            p['p_exc2vip'] = 0.5 # Proportion of VIP targeted by control input for each pathway
            p['with_VIP'] = True
        elif modeltype == 'Control2VIP_spatial':
            p['grid_on'] = True
            p['exc_som_mean'] = 0 # pA
            p['vip_arbor'] = 100 # Radius of the arbor, within which connection probability is given by p_vip2som
            p['p_exc2som'] = 1.0 # Proportion of SOM targeted by control input for each pathway
            p['p_exc2vip'] = 0.2 # Proportion of VIP targeted by control input for each pathway
            p['with_VIP'] = True
        elif modeltype == 'Control2VIP':
            p['exc_som_mean'] = 0 # pA
            p['w_vip2som_total'] = 30 # pA/Hz, total inhibitory weight received by each SOM neuron
            p['p_exc2som'] = 1 # Proportion of SOM targeted by control input for each pathway
            p['p_exc2vip'] = 0.1 # Proportion of VIP targeted by control input for each pathway
            p['with_VIP'] = True
            p['p_vip2som'] = 0.1

        p['with_PV'] = False
        p['I_PV'] = 0 # pA, current injection from PV neurons, non-negative number
        p['n_pv'] = 200
        p['p_som2pv'] = 0.8
        p['p_pv2pv'] = 0.9
        p['p_pv2pyr'] = 0.6
        p['w_som2pv_total'] = 20 # pA/Hz, total inhibitory weight received by each PV neuron from SOM neurons
        p['w_pv2pv_total']  = 30 # pA/Hz, total inhibitory weight received by each PV neuron from PV neurons
        p['w_pv2pyr_total'] = 30 # pA/Hz, total inhibitory weight received by each pyramidal soma from PV neurons

        p['modeltype'] = modeltype
        for key in outside_params:
            p[key] = outside_params[key] # overwrite the old value
        return p

    def run_vary_x(self,plot_type_list,N_rnd=1,
                   modeltype='SOM_alone', outside_params=dict(),save_name=None):

        p = self.get_p(modeltype,outside_params=outside_params)

        for key in p.keys():
            print key,
            print p[key]

        for plot_type in plot_type_list:
            var_name = plot_type
            n_som2dend = 5
            if plot_type == 'n_som':
                p_som2dend = 1-(1-p['p_som2pyr'])**(1./p['n_dend_each'])
                n_som_plot = np.round(np.arange(1,20,2)/p_som2dend/4)*4
                var_plot = n_som_plot
                p_som2dend = n_som2dend/n_som_plot
                var_name_2 = 'p_som2pyr'
                var_plot_2 = 1-(1-p_som2dend)**p['n_dend_each']
            elif plot_type == 'n_dend_each':
                n_dend_each_plot = np.array([1,2,3,5,10,15,20,25,30,35,40,50,60])
                var_plot = n_dend_each_plot
                p_som2dend = n_som2dend/p['n_som']
                var_name_2 = 'p_som2pyr'
                var_plot_2 = 1-(1-p_som2dend)**n_dend_each_plot
            elif plot_type == 'p_som2pyr':
                n_som_plot = np.array([5,16,32,64,128,160,256,512,1024])
                p_som2dend_plot = n_som2dend/n_som_plot
                p_som2pyr_plot = 1-(1-p_som2dend_plot)**p['n_dend_each']
                var_plot = p_som2pyr_plot
                var_name_2 = 'n_som'
                var_plot_2 = n_som_plot
            elif plot_type == 'n_som2dend':
                #n_som2dend_plot = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40])
                n_som2dend_plot = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
                p_som2dend = n_som2dend_plot/p['n_som']
                var_name = 'p_som2pyr'
                var_plot = 1-(1-p_som2dend)**p['n_dend_each']
            elif plot_type == 'r_som_max':
                var_plot = np.array([0,1,2,3,4,5,6,7,8,9,10])
            elif plot_type == 'n_vip':
                var_plot = np.array([10,30,50,70,100,125,150,200])
            elif plot_type == 'p_vip2som':
                var_plot = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
                #var_plot = np.linspace(0.1,0.9,20)
            elif plot_type == 'n_path':
                var_plot = np.array([2,3,4,5,6,7,8,9,10])
            elif plot_type == 'disinh_prop':
                var_plot = np.array([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
            elif plot_type == 'inh_threshold':
                var_plot = np.array([0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
            elif plot_type == 'g_exc':
                var_plot = np.array([5,10,15,20,25,30,35,40])
            elif plot_type == 'g_total':
                var_plot = np.array([10,20,30,40,50,60,70,80])
            elif plot_type == 'p_exc2som':
                var_plot = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            elif plot_type ==  'p_exc2vip':
                var_plot = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
                #var_plot = np.linspace(0.1,1.0,21)
            elif plot_type == 'exc_som_mean':
                var_plot = np.array([0,50,100,150,200,250])
            elif plot_type == 'r_vip_mean':
                #var_plot = np.array([0,10,20,30])
                var_plot = np.array([0,2,4,6,8,10])
            elif plot_type == 'vip_arbor':
                var_plot = np.array([50,100,150,200,250,300])
            elif plot_type == 'I_PV':
                var_plot = np.concatenate((np.array([0,50,60,63,65,66]),
                                           np.linspace(66.1,66.25,20),
                                           np.array([66.5,67,67.5,68,69])))
                var_plot = np.arange(0,91,10)
            else:
                IOError('Unknown Plot Type')
            

            res = dict()
            gs_plot = 0
            gs_low_plot = 0
            gs_high_plot = 0

            gs_plot_nonnaive = 0
            gs_low_plot_nonnaive = 0
            gs_high_plot_nonnaive = 0

            for i_rnd in range(N_rnd):
                print 'Varying ' + plot_type + ' Naively'
                gs_plot_temp, gs_low_plot_temp, gs_high_plot_temp = self.vary_x(p,var_name,var_plot)
                gs_plot += gs_plot_temp/N_rnd
                gs_low_plot += gs_low_plot_temp/N_rnd
                gs_high_plot += gs_high_plot_temp/N_rnd

                if plot_type in ['n_som','n_dend_each','p_som2pyr']:
                    print 'Varying ' + plot_type + ' Non-naively'
                    gs_plot_nonnaive_temp, gs_low_plot_nonnaive_temp, gs_high_plot_nonnaive_temp = self.vary_x2(p,var_name,var_plot,var_name_2,var_plot_2)
                    gs_plot_nonnaive += gs_plot_nonnaive_temp/N_rnd
                    gs_low_plot_nonnaive += gs_low_plot_nonnaive_temp/N_rnd
                    gs_high_plot_nonnaive += gs_high_plot_nonnaive_temp/N_rnd

            res['gs_plot'] = gs_plot
            res['gs_low_plot'] = gs_low_plot
            res['gs_high_plot'] = gs_high_plot
            res['var_plot'] = var_plot

            res['var_plot_nonnaive'] = var_plot
            res['gs_plot_nonnaive'] = gs_plot_nonnaive
            res['gs_low_plot_nonnaive'] = gs_low_plot_nonnaive
            res['gs_high_plot_nonnaive'] = gs_high_plot_nonnaive


            if plot_type == 'n_som2dend':
                res['var_plot'] = n_som2dend_plot


            res['base_param'] = p

            if save_name is None:
                save_name = 'GatingSelectivity_'+p['modeltype']+'vary_'+plot_type
            with open(self.datapath + save_name + self.version,'wb') as f:
                pickle.dump(res,f)

    def get_x_info(self,plot_type):
        if plot_type == 'n_som':
            x_label = r'$N_{\mathrm{SOM}}$'
            xticks = [0,200,400,600]
        elif plot_type == 'n_dend_each':
            x_label = r'$N_{dend}$'
            xticks = [0,20,40,60]
        elif plot_type == 'n_som2dend':
            x_label = r'$N_{\mathit{SOM}\rightarrow dend}$'
            xticks = [0,5,10,15,20]
        elif plot_type == 'p_som2pyr':
            x_label = r'$P_{\mathrm{SOM}\rightarrow pyr}$'
            xticks = [0,0.2,0.4,0.6,0.8,1.0]
        elif plot_type == 'r_som_max':
            x_label = r'SOM Dynamic Range (Hz)'
            xticks = [0,5,10]
        elif plot_type == 'n_path':
            x_label = r'$N_{\mathrm{path}}$'
            xticks = [0,5,10]
        elif plot_type == 'disinh_prop':
            x_label = 'Proportion of \n suppressed SOM cells'
            xticks = [0,0.5,1]
        elif plot_type == 'inh_threshold':
            x_label = 'Inhibition threshold (nS) \n for excitation'
            xticks = [0,2,4]
        elif plot_type == 'g_exc':
            x_label = 'Maximum Excitation (nS)'
            xticks = [0,20,40]
        elif plot_type == 'g_total':
            x_label = r'Total $g_{\mathit{SOM}\rightarrow dend}$ (nS)'
            xticks = [0,20,40,60,80]
        elif plot_type == 'n_vip':
            x_label = r'$N_{\mathrm{VIP}}$'
            xticks = [0,100,200]
        elif plot_type == 'p_vip2som':
            x_label = r'$P_{\mathrm{VIP}\rightarrow \mathrm{SOM}}$'
            xticks = [0,0.5,1]
        elif plot_type == 'p_exc2som':
            x_label = 'Proportion of SOM cells \n targeted by control'
            xticks = [0,0.5,1]
        elif plot_type == 'p_exc2vip':
            #x_label = 'Proportion of VIP cells \n targeted by control'
            x_label = 'Proportion of neurons \n targeted by control'
            xticks = [0,0.5,1]
        elif plot_type == 'vip_arbor':
            x_label = r'VIP Arbor Radius ($\mu m$)'
            xticks = [0,100,200,300]
        elif plot_type == 'r_vip_mean':
            x_label = 'Mean VIP firing (Hz)'
            xticks = [0,5,10]
        elif plot_type == 'I_PV':
            x_label = 'Somatic inhibition (pA)'
            xticks = [0,40,80]
        elif plot_type == 'w_som2pv':
            x_label = r'$w_{\mathit{SOM}\rightarrow \mathit{PV}}$ (pA/Hz)'
            xticks = [0,0.05,0.1,0.15]
        else:
            IOError('Unknown plot type')

        if plot_type in ['p_exc2vip','p_exc2som','p_vip2som','p_som2pyr','disinh_prop']:
            xticklabels = ['0']+['%0.1f' % x for x in xticks[1:-1]]+['1']
        elif plot_type in ['w_som2pv']:
            xticklabels = ['0'] + ['{:0.2f}'.format(x) for x in xticks[1:]]
        else:
            xticklabels = ['%d' % x for x in xticks]

        return x_label, xticks, xticklabels

    def plot_vary_x(self,plot_type,set_yticklabels=False,fighandle=None,
                    plot_color=None,savenew=True,modeltype='SOM_alone'):
        if fighandle is None:
            fig = MyFigure(figsize=(1.7,1.5))
        else:
            fig = fighandle

        with open(self.datapath+'GatingSelectivity_'+modeltype+'vary_'+plot_type+ self.version,'rb') as f:
            res = pickle.load(f)

        if plot_color is None:
            if plot_type in ['n_som','n_dend_each','p_som2pyr']:
                #plot_color=np.array([77,175,74])/255.
                plot_color = np.array([152,78,163])/255.
            elif plot_type is 'p_exc2vip':
                plot_color = np.array([102,230,0])/255
            else:
                plot_color=np.array([228,26,28])/255.

        pl=fig.addplot([0.3,0.35,0.6,0.6])
        x_label, xticks, xticklabels = self.get_x_info(plot_type)
        if plot_type in ['n_som','n_dend_each','p_som2pyr']:
            p1, = pl.plot(res['var_plot_nonnaive'],res['gs_plot_nonnaive'],color=plot_color,solid_capstyle="butt")
            #pl.ax.fill_between(res['var_plot_nonnaive'],res['gs_low_plot_nonnaive'],res['gs_high_plot_nonnaive'],facecolor=plot_color,alpha=0.2,edgecolor='white')
        else:
            p1, = pl.plot(res['var_plot'],res['gs_plot'],color=plot_color,solid_capstyle="butt")
            pl.ax.fill_between(res['var_plot'],res['gs_low_plot'],res['gs_high_plot'],facecolor=plot_color,alpha=0.2,edgecolor='white')

        pylab.xlabel(x_label)
        pl.ax.set_ylim((0,1.0))
        _=pl.ax.set_yticks([0,0.5,1.0])
        if set_yticklabels:
            _=pl.ax.set_yticklabels(['0','0.5','1'])
            pylab.ylabel('Gating selectivity')
        else:
            _=pl.ax.set_yticklabels(['','',''])
        if plot_type == 'n_som':
            pl.ax.set_xlim((xticks[0],xticks[-1]+30))
        else:
            pl.ax.set_xlim((xticks[0],xticks[-1]))
        _=pl.ax.set_xticks(xticks)
        _=pl.ax.set_xticklabels(xticklabels)

        '''
        if plot_type in ['n_som','n_dend_each','p_som2pyr']:
            p2, = pl.plot(res['var_plot_nonnaive'],res['gs_plot_nonnaive'],color=np.array([152,78,163])/255.)
        '''
        if savenew:
            fig.save(self.figpath+'GatingSelectivity_'+modeltype+'_'+plot_type + self.version)

        return pl

    def plot_vary_p_exc2neurons(self):
        fig = MyFigure(figsize=(1.7,1.5))
        self.plot_vary_x('p_exc2som',modeltype='Control2VIPSOM',set_yticklabels=False,fighandle=fig,plot_color=np.array([46,186,255])/255,savenew=False)
        pl = self.plot_vary_x('p_exc2vip',modeltype='Control2VIPSOM',set_yticklabels=True,fighandle=fig,plot_color=np.array([102,230,0])/255,savenew=False)
        pl.ax.set_xlabel('Proportion of neurons \n targeted by control')
        fig.save(self.figpath+'GatingSelectivity_p_exc2neuron' + self.version)

    def run_n_som2dend_mechanism(self):
        p = self.get_p('SOM_alone')

        n_som2dend_plot = np.array([1,2,3,4,5,6,7,8,9,10])
        p_som2dend = n_som2dend_plot/p['n_som']
        var_name = 'p_som2pyr'
        var_plot = 1-(1-p_som2dend)**p['n_dend_each']

        inh1_list = list()
        inh2_list = list()
        vDend_on_list = list()
        vDend_off_list = list()
        gs_list_list = list()
        for var in var_plot:
            p[var_name] = var
            result = self.get_gatingselectivity(p)
            inh1 = result['Inh2dend_list'][0]
            inh2 = result['Inh2dend_list'][1]
            exc1 = self.get_exc(p,inh1)
            exc2 = self.get_exc(p,inh2)
            vDend_on = dend_IO(exc1, inh1)
            vDend_off = dend_IO(exc2, inh1)
            inh1_list.append(inh1)
            inh2_list.append(inh2)
            vDend_on_list.append(vDend_on)
            vDend_off_list.append(vDend_off)
            gs_list_list.append(result['gs_list'])

        res = dict()
        res['n_som2dend_plot'] = n_som2dend_plot
        res['inh1_list'] = inh1_list
        res['inh2_list'] = inh2_list
        res['vDend_on_list'] = vDend_on_list
        res['vDend_off_list'] = vDend_off_list
        res['gs_list_list'] = gs_list_list
        res['param'] = p
        with open(self.datapath+'run_n_som2dend_mechanism'+ self.version,'wb') as f:
            pickle.dump(res,f)

    def plot_n_som2dend_mechanism(self,th = 4.0):
        with open(self.datapath+'run_n_som2dend_mechanism'+ self.version,'rb') as f:
            res = pickle.load(f)

        one_disinhibited_list = list()
        both_disinhibited_list = list()
        onlyone_disinhibited_list = list()
        variance_list = list()
        for inh1,inh2 in zip(res['inh1_list'],res['inh2_list']):
            disinhibited1 = inh1<th
            disinhibited2 = inh2<th
            both_disinhibited = [a and b for a,b in zip(disinhibited1,disinhibited2)]
            onlyone_disinhibited = [a and not b for a,b in zip(disinhibited1,disinhibited2)]
            one_disinhibited_list.append(np.sum(disinhibited1)/len(inh1))
            both_disinhibited_list.append(np.sum(both_disinhibited)/len(inh1))
            onlyone_disinhibited_list.append(np.sum(onlyone_disinhibited)/len(inh1))
            variance_list.append(np.std(inh1))

        one_disinhibited_list = np.array(one_disinhibited_list)
        both_disinhibited_list = np.array(both_disinhibited_list)
        onlyone_disinhibited_list = np.array(onlyone_disinhibited_list)

        fig = MyFigure(figsize=(1.7,1.5))

        pl=fig.addplot([0.3,0.35,0.6,0.6])

        '''
        pl.plot(res['n_som2dend_plot'],-res['n_som2dend_plot']*np.log(2))
        #pl.plot(res['n_som2dend_plot'],-2*res['n_som2dend_plot']*np.log(2))
        pl.plot(res['n_som2dend_plot'],np.log(one_disinhibited_list))
        pl.plot(res['n_som2dend_plot'],np.log(1-both_disinhibited_list))
        pl.plot(res['n_som2dend_plot'],[np.log(1./res['param']['n_dend_each'])]*len(res['n_som2dend_plot']))
        '''
        pl.plot(res['n_som2dend_plot'],one_disinhibited_list,label='P_disinh')
        pl.plot(res['n_som2dend_plot'],1-both_disinhibited_list,label='1-P_both')
        #pl.plot(res['n_som2dend_plot'],onlyone_disinhibited_list,label='P_onlyone_disinh')
        x_label, xticks, xticklabels = self.get_x_info('n_som2dend')
        pylab.xlabel(x_label)
        #pl.ax.set_ylim((0,1.0))
        #_=pl.ax.set_yticks([0,0.5,1.0])
        #_=pl.ax.set_yticklabels(['0','0.5','1'])
        pylab.ylabel('Proportion')
        leg = pylab.legend(loc=1, bbox_to_anchor=[1.2, 0.7], frameon=False)

        pl.ax.set_xlim((0,10))

        _=pl.ax.set_xticks([0,5,10])
        _=pl.ax.set_xticklabels(['0','5','10'])

        fig.save(self.figpath+'plot_n_som2dend_mechanism'+self.version)
        fig.close()

        fig = MyFigure(figsize=(1.7,1.5))
        pl=fig.addplot([0.3,0.35,0.6,0.6])
        pl.plot(res['n_som2dend_plot'],variance_list,label='P_disinh')
        x_label, xticks, xticklabels = self.get_x_info('n_som2dend')
        pylab.xlabel(x_label)
        pylab.ylabel('Variance')
        leg = pylab.legend(loc=1, bbox_to_anchor=[1.2, 0.7], frameon=False)

        pl.ax.set_xlim((0,10))

        _=pl.ax.set_xticks([0,5,10])
        _=pl.ax.set_xticklabels(['0','5','10'])

        fig.save(self.figpath+'plot_n_som2dend_mechanism3'+self.version)
        fig.close()

        vDend_on_mean = np.array([np.mean(res['vDend_on_list'][i]) for i in xrange(len(res['vDend_on_list']))])
        vDend_off_mean = np.array([np.mean(res['vDend_off_list'][i]) for i in xrange(len(res['vDend_off_list']))])

        fig = MyFigure(figsize=(1.7,1.5))

        pl=fig.addplot([0.3,0.35,0.6,0.6])
        #pl.plot(res['n_som2dend_plot'],vDend_on_mean+70)
        #pl.plot(res['n_som2dend_plot'],vDend_off_mean+70)
        #pl.plot(res['n_som2dend_plot'],(vDend_off_mean+70)/(vDend_on_mean+70))
        pl.plot(res['n_som2dend_plot'],np.log10(vDend_on_mean+70))
        pl.plot(res['n_som2dend_plot'],np.log10(vDend_off_mean+70))
        x_label, xticks, xticklabels = self.get_x_info('n_som2dend')
        pylab.xlabel(x_label)
        #pl.ax.set_ylim((0,1.0))
        #_=pl.ax.set_yticks([0,0.5,1.0])
        #_=pl.ax.set_yticklabels(['0','0.5','1'])
        pylab.ylabel('HULU')

        pl.ax.set_xlim((0,10))

        _=pl.ax.set_xticks([0,5,10])
        _=pl.ax.set_xticklabels(['0','5','10'])

        fig.save(self.figpath+'plot_n_som2dend_mechanism2'+self.version)
        fig.close()

        #plot_item_list = ['vDend_on_list','vDend_off_list']
        plot_item_list = ['gs_list_list']

        for plot_item in plot_item_list:
            fig = pylab.figure(1,(1.5,1.5))
            #L = len(res[plot_item])
            L = 8
            step = 1
            Np = int(np.floor(L/step))
            color_hist = 'black'

            if plot_item in ['vDend_on_list','vDend_off_list']:
                bins = np.linspace(-70,-10,100)
            elif plot_item is 'gs_list_list':
                bins = np.linspace(0,1,20)

            for i in xrange(Np):
                ax = fig.add_axes((0.2,0.9-0.7/Np*(i+1),0.6,0.7/Np))
                hist, bin_edges = np.histogram(res[plot_item][i*step], bins=bins)
                #pylab.setp(patches, 'facecolor', 'green','edgecolor','none')
                print('N_som2dend is %d' % res['n_som2dend_plot'][i*step])
                print hist
                #print(res['n_som2dend_plot'][i*step])
                #pylab.axis('off')
                ax.bar(bin_edges[:-1],hist,width=bin_edges[1]-bin_edges[0],edgecolor='none')
                pylab.ylim((0,len(res[plot_item][0])))
                #pylab.xlim([-70,20])

            fig.savefig(self.figpath+'plot_'+plot_item+self.version+'.pdf',bbox_inches='tight', pad_inches = 0)
            plt.close()

    def gs_mechanism(self):
        self.version = 22
        with open(self.datapath+'GatingSelectivity_param_V%d' % self.version,'rb') as f:
            p = pickle.load(f)

        n_plot = 30
        inh1_range = np.linspace(0,7,n_plot)
        inh2_range = np.linspace(0,7,n_plot)[::-1]
        Inh1_range, Inh2_range = np.meshgrid(inh1_range,inh2_range)
        inh1 = Inh1_range.flatten()
        inh2 = Inh2_range.flatten()
        exc1 = self.get_exc(p,inh1)
        exc2 = self.get_exc(p,inh2)

        Von = dend_IO(exc1,inh1)
        Voff = dend_IO(exc2,inh1)
        Vnone = dend_IO(exc1*0,inh1)

        dVon = Von-Vnone
        dVoff = Voff-Vnone
        gs1 = (dVon-dVoff)/(dVon+dVoff+1e-9)

        Von_show = Von.reshape((n_plot,n_plot))
        Voff_show = Voff.reshape((n_plot,n_plot))


        p2 = copy.deepcopy(p)
        p2['n_som'] = 100
        result = self.get_gatingselectivity(p2)
        color='black'
        n_dend_show = 1000

        inh2dend_path0 = result['Inh2dend_list'][0][:n_dend_show]
        inh2dend_path1 = result['Inh2dend_list'][1][:n_dend_show]

        inh2dend_path0 += np.random.randn(n_dend_show)*0.05
        inh2dend_path1 += np.random.randn(n_dend_show)*0.05

        print 'N SOM-->Dend = %0.2f' % result['param']['n_som2dend']


        for V_show, V_name in zip([Von_show,Voff_show],['Von','Voff']):
            vmax = -30
            cm = 'cool'
            fig = MyFigure(figsize=(2.0,1.5))
            pl=fig.addplot([0.3,0.3,0.4,0.6])
            im = pl.ax.imshow(V_show,vmin=-70,vmax=vmax,cmap=cm,aspect=1,
                              extent=[inh1_range.min(), inh1_range.max(), inh2_range.min(), inh2_range.max()])
            pl.ax.set_xticks([0,3,6])
            pl.ax.set_yticks([0,3,6])
            pl.ax.set_xlabel('Inh 1 (nS)')
            pl.ax.set_ylabel('Inh 2 (nS)')

            pl.plot(inh2dend_path0,inh2dend_path1,
                '.',alpha=0.3,markerfacecolor=color,markeredgecolor=color,markersize=1)
            pl.ax.set_xlim((-0.5,7))
            pl.ax.set_ylim((-0.5,7))

            pl=fig.addplot([0.75,0.3,0.02,0.6])
            cb = plt.colorbar(im,ticks=[-70,-50,-30],cax=pl.ax)
            cb.set_label('Dendritic Voltage (-mV)', rotation=270,labelpad=10)
            fig.save(self.figpath+'GatingSelectivity_inh1vsinh2_'+V_name+'_V%d' % self.version)

    def generate_W_grid(self,p):
        '''
        Generate connection matrix for neurons in a two-dimensional grid
        Specifically for VIP-SOM connections
        '''
        p['p_vip2som_arbor'] = 0.6
        # Consider a grid of 400\mum * 400\mum
        # Assign locations of neurons
        p['n_vip_scale'] = 625
        p['grid_size_vip'] = 400*np.sqrt(p['n_vip_scale']/p['n_vip'])  # mu m
        p['n_vip_scale_sqrt'] = np.round(np.sqrt(p['n_vip_scale']))
        p['n_som_sqrt'] = np.floor(np.sqrt(p['n_som']))

        # x and y locations of VIP neurons, randomly drawn
        p['vip_x'] = np.tile(np.linspace(-0.5,0.5,p['n_vip_scale_sqrt']),p['n_vip_scale_sqrt'])*p['grid_size_vip']
        p['vip_y'] = np.repeat(np.linspace(-0.5,0.5,p['n_vip_scale_sqrt']),p['n_vip_scale_sqrt'])*p['grid_size_vip']
        # x and y locations of SOM neurons, randomly drawn
        p['som_x'] = np.tile(np.linspace(-0.5,0.5,p['n_som_sqrt']),p['n_som_sqrt'])*400
        p['som_y'] = np.repeat(np.linspace(-0.5,0.5,p['n_som_sqrt']),p['n_som_sqrt'])*400
        p['som_x'] = np.concatenate((p['som_x'],(np.random.rand(p['n_som']-p['n_som_sqrt']**2)-0.5)*400))
        p['som_y'] = np.concatenate((p['som_y'],(np.random.rand(p['n_som']-p['n_som_sqrt']**2)-0.5)*400))

        # Assume that each VIP only targets SOM within vicinity (vip_arbor) with probability p_vip2som_arbor
        p['W_vip2som'] = np.zeros((p['n_som'],p['n_vip_scale']))

        for i_som in xrange(p['n_som']):
            dist2vip = np.sqrt((p['som_x'][i_som]-p['vip_x'])**2+(p['som_y'][i_som]-p['vip_y'])**2)
            # Make connections if p>p_vip2som_arbor and dist<vip_arbor
            ind_vip2som_conn = np.where(dist2vip<(p['vip_arbor']))[0]
            np.random.shuffle(ind_vip2som_conn)
            ind_vip2som_conn = ind_vip2som_conn[:int(p['p_vip2som_arbor']*len(ind_vip2som_conn))]
            p['W_vip2som'][i_som,ind_vip2som_conn] = 1

        n_vip2som = np.sum(p['W_vip2som'],axis=1)
        # uIPSQ is about 0.7 pC=0.7 pA/Hz for VIP-SOM connection, Pfeffer et al. Nat Neurosci. 2012
        #syn_weight_vip2som = 10/n_vip2som
        for i_som in xrange(p['n_som']):
            p['W_vip2som'][i_som,:] = p['W_vip2som'][i_som,:]*0.7*60/n_vip2som[i_som]

        return p

    def get_r_som_grid(self,p):
        # Generate information needed for the grid
        p = self.generate_W_grid(p)

        Exc2som_list = list()
        Inh2som_list = list()
        r_som_list = list()


        for i_path in range(p['n_path']):
            Exc2som = np.zeros(p['n_som'])

            n_exc2vip = round(p['p_exc2vip']*p['n_vip_scale'])
            r_vip = np.zeros(p['n_vip_scale'])
            r_vip[:n_exc2vip] = p['r_vip_mean']*p['n_vip_scale']/n_exc2vip
            np.random.shuffle(r_vip)

            Inh2som = np.dot(p['W_vip2som'],r_vip)
            input2som = Exc2som - Inh2som + p['Exc2som0']

            r_som_list.append(self.fI_SOM(input2som))
            Exc2som_list.append(Exc2som)
            Inh2som_list.append(Inh2som)

        return r_som_list, Exc2som_list, Inh2som_list

    def run_fancyvary_somainh(self,N_rnd=1,var_name='w_som2pv'):
        modeltype='Control2VIPSOM'
        p = self.get_p(modeltype,outside_params=dict())

        res_all = dict()
        res_list = list()
        n_som2dend_list = [5,10,15]

        if var_name == 'w_som2pv':
            p['with_PV'] = True
            w_som2pv_plot_1 = np.concatenate((np.linspace(0,0.10,10),
                                              np.linspace(0.1,0.15,25)))
            w_som2pv_plot_2 = np.concatenate((np.linspace(0,0.11,11),
                                              np.linspace(0.11,0.125,15),
                                              np.linspace(0.125,0.15,5)))
            var_plot_list = np.array([w_som2pv_plot_1,w_som2pv_plot_2,w_som2pv_plot_2])

            var_name_run = 'w_som2pv_total'
            var_plot_run_list = var_plot_list*p['n_som']*p['p_som2pv']

        elif var_name == 'I_PV':
            var_plot_1 = np.concatenate((np.array([0,10,20,30,40,50,55]),
                                         np.linspace(60,80,25)))
            var_plot_dense = np.concatenate((np.array([0,25,50,55,58,60,62,64,65,66]),
                                           np.linspace(66.1,66.25,10),
                                           np.array([66.5,67,67.5,68,69,72,75,80])))
            var_plot_list = [var_plot_1,
                             var_plot_dense,
                             var_plot_dense]
            p['with_PV'] = False
            var_name_run = var_name
            var_plot_run_list = var_plot_list

        for i in range(len(n_som2dend_list)):

            p_som2dend = n_som2dend_list[i]/p['n_som']
            p['p_som2pyr'] = 1-(1-p_som2dend)**p['n_dend_each']

            for key in p.keys():
                print key,
                print p[key]

            gs_plot = 0

            for i_rnd in range(N_rnd):
                gs_plot_temp, _ , _ = self.vary_x(p,var_name_run,var_plot_run_list[i])
                gs_plot += gs_plot_temp/N_rnd

            res = dict()
            res['gs_plot'] = gs_plot
            res['var_plot'] = var_plot_list[i]
            res['base_param'] = p

            res_list.append(res)

        res_all['res_list'] = res_list
        res_all['var_plot_list'] = var_plot_list
        res_all['n_som2dend_list'] = n_som2dend_list

        with open(self.datapath + 'GatingSelectivity_'+p['modeltype']+'fancyvary_'+var_name + self.version,'wb') as f:
            pickle.dump(res_all,f)

    def plot_fancyvary_somainh(self,set_yticklabels=True,fighandle=None,savenew=True,var_name='w_som2pv'):

        with open(self.datapath + 'GatingSelectivity_Control2VIPSOMfancyvary_'+var_name + self.version,'rb') as f:
            res_all = pickle.load(f)

        if fighandle is None:
            fig = MyFigure(figsize=(1.7,1.5))
        else:
            fig = fighandle

        pl=fig.addplot([0.3,0.25,0.6,0.6])
        x_label, xticks, xticklabels = self.get_x_info(var_name)

        plot_colors = np.array([[127,39,4],[241,105,19],[253,174,107]])/255.

        for i in range(len(res_all['res_list'])):
            res = res_all['res_list'][i]
            p1, = pl.plot(res['var_plot'],res['gs_plot'],color=plot_colors[i],
                          solid_capstyle="butt",label='{:d}'.format(res_all['n_som2dend_list'][i]))

        pylab.xlabel(x_label)
        pl.ax.set_ylim((0,1.0))
        _=pl.ax.set_yticks([0,0.5,1.0])
        if set_yticklabels:
            _=pl.ax.set_yticklabels(['0','0.5','1'])
            pylab.ylabel('Gating selectivity')
        else:
            _=pl.ax.set_yticklabels(['','',''])
        pl.ax.set_xlim((xticks[0],xticks[-1]))
        _=pl.ax.set_xticks(xticks)
        _=pl.ax.set_xticklabels(xticklabels)

        pl.legend(bbox_to_anchor=[0.6, 1.3], frameon=False, title=r'$N_{\mathit{SOM}\rightarrow dend}$')

        if savenew:
            fig.save(self.figpath+'GatingSelectivity_Control2VIPSOMfancyvary_'+var_name+ self.version)

    def plot_inhexc(self):
        # Plot exc and inh distribution and relationship

        p = IC.get_p(modeltype='SOM_alone', outside_params={})

        n_som2dend_plot = np.array([5,15])
        p_som2dend = n_som2dend_plot/p['n_som']
        var_name = 'p_som2pyr'
        var_plot = 1-(1-p_som2dend)**p['n_dend_each']

        result_list = list()
        for var_val in var_plot:
            p = IC.get_p(modeltype='SOM_alone', outside_params={var_name:var_val})
            result_list.append(IC.get_gatingselectivity(p))

        n_dend_show = 1000
        fig = MyFigure(figsize=(1.5,1.5))
        pl=fig.addplot([0.2,0.2,0.7,0.7])
        for result, color in zip(result_list,[ 'blue','red']):
            pl.plot(result['Inh2dend_list'][0][:n_dend_show]+np.random.randn(n_dend_show)*0.08,
                    result['Exc2dend_list'][0][:n_dend_show]+np.random.randn(n_dend_show)*0.25,
                    '.',alpha=0.1,markerfacecolor=color,markeredgecolor=color,markersize=1)
            #pl.ax.set_xlim((-0.5,7))
            #pl.ax.set_ylim((-0.5,7))
            #_=pl.ax.set_xticks([0,7])
            #_=pl.ax.set_yticks([0,7])
            pl.xlabel('Inhibition (nS)')
            pl.ylabel('Excitation (nS)')

            #fig.save(self.figpath+'GatingSelectivity_inh1vsinh2_nsom2dend%d_V%d' % (result['param']['n_som2dend'],self.version))
        pl.ax.set_xticks([0,2,4,6,8])
        pl.ax.set_yticks([0,5,10,15,20,25])
    
        colors = ['blue','red']
        fig = MyFigure(figsize=(1.5,1.5))
        for i in [1,0]:
            result = result_list[i]
            pl=fig.addplot([0.2,0.2+0.4*i,0.7,0.3])
            _ = plt.hist(result['Inh2dend_list'][0],bins=100)
            #pl.ax.spines['left'].set_visible(False)
            #pl.ax.get_yaxis().set_visible(False)
            pl.xlim([-0.5,8.5])
            pl.ax.set_xticks([0,2,4,6,8])
            if i == 1:
                pl.ax.set_xticklabels([])
            else:
                pl.xlabel('Inhibition (nS)')

        fig = MyFigure(figsize=(1.5,1.5))
        pl=fig.addplot([0.2,0.2,0.7,0.7])
        result = result_list[0]
        for i_path, color in zip([0,1],['blue','red']):
            pl.plot(result['Inh2dend_list'][0][:n_dend_show]+np.random.randn(n_dend_show)*0.08,
                    result['Exc2dend_list'][i_path][:n_dend_show]+np.random.randn(n_dend_show)*0.25,
                    '.',alpha=0.1,markerfacecolor=color,markeredgecolor=color,markersize=1)
        pl.xlabel('Inhibition (nS)')
        pl.ylabel('Excitation (nS)')
        pl.ax.set_xticks([0,2,4,6,8])
        pl.ax.set_yticks([0,5,10,15,20,25])
