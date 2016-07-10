'''2014-12-09
Learning pathway gating, and also illustrate pathway gating
'''

from __future__ import division
import time
import os
import sys
import re
import pickle
import numpy.random
import scipy as sp
import scipy.optimize
import scipy.signal
import bisect
import random as pyrand
import brian_no_units
from brian import *
import MultiCompartmentalModel as MCM
from figtools import MyFigure


colorMatrix = array([[127,205,187],
                    [65,182,196],
                    [29,145,192],
                    [34,94,168],
                    [37,52,148],
                    [8,29,88]])/255.

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

        #self.params = read_params(paramsfile)

    def run_WeightChangevsrE(self,restore=True):
        '''
        Run the neuron under different excitatory input rate and inhibition level,
        in order to calculate the calcium dynamics, and therefore the weight change
        '''
        p = dict()
        p['dt'] = 0.1*ms
        p['record_dt'] = 0.5*ms
        p['num_DendEach'] = 1 # For clamped soma, dendrite number does not matter
        p['gGABA'] = 4.0*nS
        p['gNMDA'] = 2.5*nS
        p['pre_rates'] = linspace(0,50,21)*Hz
        p['n_rate'] = len(p['pre_rates'])
        p['n_rep'] = 1
        num_soma = p['n_rep']*p['n_rate']
        p['num_input'] = 15
        #p['post_rate'] = 0*Hz
        #dend_inh_rates = array([0,50,100])*Hz
        p['post_rate'] = 10*Hz
        dend_inh_rates = array([25,75,150])*Hz

        p['dend_inh_rates'] = dend_inh_rates

        runtime = 60*second

        result_list = list()
        for dend_inh_rate in dend_inh_rates:
            model = MCM.Model(paramsfile=self.paramsfile,
                              eqsfile=self.eqsfile,outsidePara=p)
            # mimic the in vivo condition by reducing coupling weight, and
            # clamping somatic membrane potential around -60mV, but not providing direct EI input to soma
            '''
            model.make_model(num_soma = num_soma, condition = 'invivo')
            model.rate_experiment(num_input=p['num_input'], pre_rates = p['pre_rates'],
                            post_rate = p['post_rate'], dend_inh_rate=dend_inh_rate)
            '''
            model.make_model_dendrite_only(num_soma=num_soma, condition = 'invivo',
                           clamped_dendvolt=-60*mV)
            model.rate_experiment(num_input=p['num_input'], pre_rates = p['pre_rates'],
                            post_rate = p['post_rate'], dend_inh_rate=dend_inh_rate)

            model.make_network()
            model.reinit()
            net = Network(model)
            net.run(runtime,report='text')
            mon = model.monitor

            tplot = mon['MvDend'].times

            p['dend_inh_rate'] = dend_inh_rate
            result = dict()
            result['params'] = p
            result['vDend'] = mon['MvDend'].values[:,tplot>500*ms]
            result['NMDACasyn'] = mon['MNMDACasyn'].values[:,tplot>500*ms]
            result['bAPCa'] = mon['MbAPCa'].values[:,tplot>500*ms]
            result['dend_inh_rate'] = dend_inh_rate
            result_list.append(result)

        if restore:
            with open(self.datapath+'resultlist_varyrate_postr%d' % p['post_rate']+self.version,'wb') as f:
                pickle.dump(result_list,f)


    def plot_WeightChangevsrE(self):
        '''
        Plot weight change when varying excitatory input rate
        '''
        with open(self.datapath+'resultlist_varyrate_postr10','rb') as f:
            result_list = pickle.load(f)

        with open(self.datapath+'learning_para_2014-11-10_0','rb') as f:
            para = pickle.load(f)
        q = para.copy()
        # in vivo change (see Higgins, Graupner, Brunel 2014)
        #q['NMDA_scaling'] = q['NMDA_scaling']*1.5/2
        #q['bAP_scaling'] = q['bAP_scaling']/5

        vDend_mean_list = list()
        w_post_list = list()
        AlphaP_list = list()
        AlphaD_list = list()
        for result in result_list:
            p = result['params']
            vDend = result['vDend']
            NMDACasyn = result['NMDACasyn']
            bAPCa = result['bAPCa']

            vDend_mean = vDend.mean(axis=1)
            vDend_mean_list.append(vDend_mean)


            num_syn = NMDACasyn.shape[0]
            num_soma = bAPCa.shape[0]
            synsomaratio = num_syn//num_soma

            bAPCasyn = bAPCa.repeat(synsomaratio,axis=0)
            CaTrace = NMDACasyn*q['NMDA_scaling'] + bAPCasyn*q['bAP_scaling']

            w_post = zeros(num_syn)
            AlphaP_plot = zeros(num_syn)
            AlphaD_plot = zeros(num_syn)

            n_eachrate = num_syn//p['n_rate']
            repeat_times = 5
            record_dt = p['record_dt']
            for i_syn in xrange(num_syn):
                CaTraceSorted = sort(CaTrace[i_syn])

                AlphaP = MCM.crossthresholdtime(CaTraceSorted,q['thetaP'])*record_dt*repeat_times
                AlphaD = MCM.crossthresholdtime(CaTraceSorted,q['thetaD'])*record_dt*repeat_times

                wpre = 1
                wpost_syn = MCM.SynapseChange(wpre,AlphaP,AlphaD,q)
                w_post[i_syn] = wpost_syn
                AlphaP_plot[i_syn] = AlphaP
                AlphaD_plot[i_syn] = AlphaD

            w_post = w_post.reshape(p['n_rate'],n_eachrate)
            w_post = w_post.mean(axis=1)
            w_post_list.append(w_post)
            AlphaP_list.append(AlphaP_plot)
            AlphaD_list.append(AlphaD_plot)

        dend_inh_rates = p['dend_inh_rates']


        colorMatrix = array([[65,182,196],
                            [34,94,168],
                            [8,29,88]])/255

        fig=MyFigure(figsize=(2.5,2.5))
        pl = fig.addplot(rect=[0.2,0.25,0.7,0.7])
        plotlist = list()
        i = 0
        for vDend_mean in vDend_mean_list:
            ax, = pl.plot(p['pre_rates'],vDend_mean/mV,color=colorMatrix[i,:])
            plotlist.append(ax)
            xlabel('Pre-synaptic rate (Hz)')
            ylabel('Mean Dendritic Voltage (mV)')
            i += 1
        pl.ax.set_xticks([0,20,40])
        pl.ax.set_ylim((-70,-10))
        pl.ax.set_yticks([-70,-50,-30,-10])
        leg=legend(plotlist,['%d' % r for r in dend_inh_rates],title='Inhibition (Hz)',loc=2)
        leg.draw_frame(False)
        #fig.save('figures/RateVsInhibition'+self.version)



        fig=MyFigure(figsize=(2,2))
        pl = fig.addplot(rect=[0.2,0.2,0.7,0.7])
        pl.plot([0,max(p['pre_rates'])],[1,1],color=array([189,189,189])/255)
        i = 0
        plotlist = list()
        for w_post in w_post_list:
            ax, = pl.plot(p['pre_rates'],w_post,color=colorMatrix[i,:])
            plotlist.append(ax)
            xlabel('Pre-synaptic rate (Hz)')
            ylabel('Weight change')
            i += 1
        pl.ax.set_xticks([0,20,40])
        pl.ax.set_ylim((0,3))
        pl.ax.set_yticks([0,1,2,3])
        leg=legend(plotlist,['%d' % r for r in dend_inh_rates],title='Inhibition (Hz)',loc=2)
        leg.draw_frame(False)
        #title('post rate = %d Hz' % p['post_rate'])
        fig.save(self.figpath+'WeightChangevsrE_postr%d' % p['post_rate']+self.version)


    def run_basic_pathway_gating(self,mode,run_slice=False):

        p = dict()
        if run_slice:
            np = 21
            x = concatenate((linspace(-2.4,2.4,np),ones(np)))
            input_rate = exp(-x**2)
            input_rate = input_rate/input_rate.max()*40
            p['pre_rates'] = [input_rate,input_rate[::-1]]
        else:
            np = 11
            x = linspace(0,2.4,np)
            [X,Y] = meshgrid(x,x)

            input_rate = exp(-X**2)
            input_rate = input_rate/input_rate.max()*40
            p['pre_rates'] = [input_rate.flatten(),input_rate.T.flatten()]

        p['np'] = np
        p['bkg_inh_rate'] = 5*Hz
        p['dend_inh_rate'] = 30*Hz
        NMDA_prop = 1

        if mode is 'specific':
            p['w_input'] = (1,0)
            p['num_input'] = 15
        elif mode is 'specific_closed':
            p['w_input'] = (1,0)
            p['num_input'] = 15
            p['bkg_inh_rate'] = 35*Hz
            p['dend_inh_rate'] = 0*Hz
        elif mode is 'prelearning':
            p['w_input'] = (1,1)
            p['num_input'] = 8 # Chosen such that pre and post learning have same total strength
        elif mode is 'postlearning':
            p['w_input'] = (2.75,0.47)
            p['num_input'] = 5
        elif mode is 'AMPA':
            p['w_input'] = (1,0)
            p['num_input'] = 15
            NMDA_prop = 0
        elif mode is 'NMDAAMPA':
            p['w_input'] = (1,0)
            p['num_input'] = 15
            NMDA_prop = 0.5
        else:
            ValueError('Unknown mode')
        num_soma = len(p['pre_rates'][0])
        p['num_soma'] = num_soma
        p['runtime'] = 60*second

        p['dt'] = 0.5*ms
        p['record_dt'] = 1.0*ms
        p['num_DendEach'] = 10

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

        p['firing_rate_plot'] = firing_rate_plot

        if run_slice:
            with open(self.datapath+'basic_pathway_gating_response_slice_'+mode+self.version,'wb') as f:
                pickle.dump(p,f)
            with open(self.datapath+'basic_pathway_gating_response_slice_'+mode+'_latest','wb') as f:
                pickle.dump(p,f)
        else:
            with open(self.datapath+'basic_pathway_gating_response_'+mode+self.version,'wb') as f:
                pickle.dump(p,f)
            with open(self.datapath+'basic_pathway_gating_response_'+mode+'_latest','wb') as f:
                pickle.dump(p,f)


    def plot_basic_pathway_gating(self,mode,plot_ylabel=True):

        with open(self.datapath+'basic_pathway_gating_response_'+mode+'_latest','rb') as f:
            p = pickle.load(f)

        response = p['firing_rate_plot'].reshape((p['np'],p['np']))
        #print response

        temp = response[:,::-1]
        temp = temp[:,:-1]
        temp2 = concatenate([temp,response],axis=1)
        temp = temp2[::-1,:]
        temp = temp[:-1,:]
        response_exp = concatenate([temp,temp2],axis=0)

        rnone = response[-1,-1]
        ron = response[-1,0]
        roff = response[0,-1]
        dron = ron-rnone
        droff = roff-rnone
        print 'Gating selectivity = %0.2f' % ((dron-droff)/(dron+droff))


        #response_exp = p['firing_rate_plot'].reshape((p['np'],p['np']))

        if mode is 'specific_closed':
            with open(self.datapath+'basic_pathway_gating_response_'+'specific'+'_latest','rb') as f:
                p2 = pickle.load(f)
            maxr = p2['firing_rate_plot'].max()
        elif mode is 'prelearning':
            with open(self.datapath+'basic_pathway_gating_response_'+'postlearning'+'_latest','rb') as f:
                p2 = pickle.load(f)
            maxr = p2['firing_rate_plot'].max()
        else:
            maxr = response_exp.max()
        if mode in ['prelearning','postlearning']:
            minr = 4
        else:
            minr = 0
        fs = 7
        fig = plt.figure(1,(1.7,1.7))
        ax = fig.add_axes([0.1,0.2,0.6,0.6])
        im0 = ax.imshow(response_exp, interpolation='none',cmap='hot', vmin=minr, vmax=maxr)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if plot_ylabel:
            cbar = plt.colorbar(im0, format="%d", cax = axes([0.75, 0.2, 0.05, 0.6]),ticks=(minr,(maxr//5)*5))
            cbar.set_label('Output rate (Hz)',rotation=270,fontsize=fs)
        savefig(self.figpath+'basic_pathway_gating_'+mode+self.version+'.pdf',format='pdf')


    def plot_basic_pathway_gating_slice(self,mode,plot_ylabel=True):
        with open(self.datapath+'basic_pathway_gating_response_slice_'+mode+'_latest','rb') as f:
            p = pickle.load(f)
        #print p['firing_rate_plot']
        fig=MyFigure(figsize=(1.8,0.75))
        pl = fig.addplot(rect=[0.25,0.15,0.55,0.65])
        #pl.plot(p['firing_rate_plot'][:p['np']],color=colorMatrix[1],label='%d' % p['bkg_inh_rate'])
        #pl.plot(p['firing_rate_plot'][p['np']:],color=colorMatrix[5],label='%d' % (p['bkg_inh_rate']+p['dend_inh_rate']))
        pl.plot(p['firing_rate_plot'][:p['np']],color=colorMatrix[1],label='1')
        pl.plot(p['firing_rate_plot'][p['np']:],color=colorMatrix[5],label='2')

        #xlabel('Pathway 1 stimulus')
        #xticks([p['np']//2],['preferred'])
        xticks([p['np']//2],[''])

        if mode in ['prelearning','postlearning']:
            maxr = 25
        else:
            maxr = p['firing_rate_plot'].max()
        pl.ax.set_ylim((0,maxr))
        pl.ax.set_yticks([0,maxr//5*5])
        if plot_ylabel:
            ylabel('Output rate (Hz)')
            leg = legend(title='Open gate',loc=1, bbox_to_anchor=[1.0, 1.35], frameon=False)
        else:
            pl.ax.set_yticklabels(['',''])
        fig.save(self.figpath+'basic_pathway_gating_slice_'+mode+self.version)


        ron = np.max(p['firing_rate_plot'][:p['np']]) - np.min(p['firing_rate_plot'][:])
        roff = np.max(p['firing_rate_plot'][p['np']:]) - np.min(p['firing_rate_plot'][:])

        print 'Gating selectivity is %0.2f' % ((ron-roff)/(ron+roff))

#os.system('say "your program has finished"')

