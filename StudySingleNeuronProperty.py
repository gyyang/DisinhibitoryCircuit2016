'''
2014-12-04
# These files study the property of single neurons
# Numerical issue is encountered when dt = 0.1ms
'''

from __future__ import division
import time
import os
import re
import pickle
import numpy.random
import scipy as sp
import random as pyrand
import brian_no_units
from brian import *
import MultiCompartmentalModel as MCM
from figtools import MyFigure, MySubplot



colorMatrix = array([[127,205,187],
                    [65,182,196],
                    [29,145,192],
                    [34,94,168],
                    [37,52,148],
                    [8,29,88]])/255

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


    def run_NMDAspike(self):
        '''
        Plot the NMDA spike with increasing number of activated synapses
        '''
        #remakeFig=True
        num_input_list = [30,35,40,45,50]
        p = dict()
        p['num_DendEach']=10
        p['dt'] = 0.02*ms
        p['record_dt'] = 0.5*ms
        p['gNMDA'] = 2.5*nS
        inputtime = 50*ms
        vDend_list = list()
        vSoma_list = list()
        for num_input in num_input_list:
            model = MCM.Model(paramsfile=self.paramsfile, eqsfile=self.eqsfile,outsidePara=p)
            model.make_model(num_soma = 1)
            model.spike_train_experiment(pre_time = inputtime, post_times=None, num_input=num_input)
            model.make_network()
            model.reinit()
            net = Network(model)
            net.run(200*ms,report='text')
            mon = model.monitor

            vDend_list.append(mon['MvDend'][0]/mvolt)
            vSoma_list.append(mon['MvSoma'][0]/mvolt)

        res = dict()
        res['times'] = (mon['MvDend'].times-inputtime)/ms
        res['vDend_list'] = vDend_list
        res['vSoma_list'] = vSoma_list
        with open(self.datapath+'NMDAspike_latest.pkl','wb') as f:
            pickle.dump(res,f)

    def plot_NMDAspike(self,remakeFig=False):
        with open(self.datapath+'NMDAspike_latest.pkl','rb') as f:
            res = pickle.load(f)

        fig = MyFigure(figsize=(1.4,0.8))
        pd=fig.addplot([0.3,0.4,0.65,0.55])
        for j in xrange(len(res['vDend_list'])):
            plot(res['times'],res['vDend_list'][j],color='black')
        pd.xlabel('Time (ms)',labelpad=3)
        pd.ylabel('$V_D$ (mV)',labelpad=2)
        pd.ax.set_yticks([-70,-20])
        pd.ax.set_xticks([0,100])
        pd.xaxis.set_tick_params(pad=0.)
        pd.yaxis.set_tick_params(pad=0.)
        if remakeFig:
            fig.save(self.figpath+'plot_NMDAspike_Dend'+self.version)

        fig = MyFigure(figsize=(1.4,0.7))
        pd=fig.addplot([0.3,0.3,0.65,0.65])
        for j in xrange(len(res['vDend_list'])):
            plot(res['times'],res['vSoma_list'][j],color='black')
        pd.xlabel('Time (ms)',labelpad=3)
        ylabel('$V_S$ (mV)',labelpad=0)
        pd.ax.set_yticks([-70,-60])
        pd.ax.set_xticks([0,100])
        pd.xaxis.set_tick_params(pad=0.)
        pd.yaxis.set_tick_params(pad=0.)
        if remakeFig:
            fig.save(self.figpath+'plot_NMDAspike_Soma'+self.version)
        show()


    def run_EvsDI(self,denseinh=False, load_previous = True, name = 'NMDA',
                  inh_list = array([0,5,10,20,30])*Hz, rate_range=60):
        '''
        Get mean dendritic voltage as a function of excitatory input rate,
        and inhibitory rate
        '''


        n_rep = 10
        n_rate = 31
        pre_rates = linspace(0,rate_range,n_rate)*Hz
        pre_rates_exp = pre_rates.repeat(n_rep)
        #inh_list = array([0,5,10,20,30])*Hz
        p = dict()
        p['nInhValue'] = len(inh_list)

        if denseinh:
            di = 'denseinh_'
            k = 10
        else:
            di = ''
            k = 1

        filename = self.datapath+'EvsDI_'+di+name+('_range%d' % rate_range)+'_latest.pkl'

        if load_previous:
            with open(filename,'rb') as f:
                res = pickle.load(f)
        else:
            res = dict()
            res['vDend_mean'] = dict()

        for i_inh in xrange(p['nInhValue']):
            p['bkg_inh_rate'] = inh_list[i_inh]*k
            p['dend_inh_rate'] = 0*Hz
            p['pre_rates'] = pre_rates
            p['w_input'] = 1
            p['num_input'] = 15

            num_soma = len(pre_rates_exp)
            p['num_soma'] = num_soma
            p['runtime'] = 12*second
            #p['runtime'] = 1*second
            #p['dt'] = 0.02*ms
            p['dt'] = 0.2*ms
            p['record_dt'] = p['dt']
            # Here the number of dendrites is conveniently set to 1, since
            # the somatic voltage is clamped at -60mV
            p['num_DendEach'] = 1

            model = MCM.Model(paramsfile=self.paramsfile, eqsfile=self.eqsfile,outsidePara=p)
            model.make_model_dendrite_only(num_soma = num_soma,clamped_somavolt=-60*mV,condition='invivo')
            model.make_dend_bkginh(inh_rates=p['bkg_inh_rate'])
            if name is 'NMDA':
                model.single_pathway_gating_experiment(pre_rates_exp, num_input=p['num_input'],
                                                       w_input = p['w_input'])
            elif name is 'AMPA':
                model.single_pathway_gating_experiment_AMPA(pre_rates_exp, num_input=p['num_input'])
            elif name is 'NMDA_nonsatur':
                p['w_input'] = 0.2
                model.single_pathway_gating_experiment_NMDAnonsatur(pre_rates_exp, num_input=p['num_input'],
                                                       w_input = p['w_input'])
            else:
                ValueError('Unknown case.')

            model.make_network()
            model.reinit()
            net = Network(model)
            net.run(p['runtime'],report='text')

            mon = model.monitor

            plottime = mon['MvDend'].times
            vDend = mon['MvDend'].values
            vDend = vDend[:,plottime>500*ms]
            vDend_mean = vDend.mean(axis=1)/mV
            vDend_mean = vDend_mean.reshape((n_rate,n_rep))
            vDend_mean = vDend_mean.mean(axis=1)
            # Update previous data
            if inh_list[i_inh] in res['vDend_mean'].keys():
                res['vDend_mean'][inh_list[i_inh]] = (res['vDend_mean'][inh_list[i_inh]]/2+
                vDend_mean/2)
            else:
                res['vDend_mean'][inh_list[i_inh]] = vDend_mean

        res['pre_rates'] = pre_rates
        res['params'] = p


        with open(filename,'wb') as f:
            pickle.dump(res,f)


    def plot_EvsDI(self,name = 'NMDA',denseinh = False,bothinh = True,rate_range=60):
        '''
        plot mean dendritic voltage as a function of excitatory input rate,
        and inhibitory rate
        '''

        if denseinh:
            di = 'denseinh_'
            k = 10
        else:
            di = ''
            k = 1

        name_append = di+name+('_range%d' % rate_range)+self.version

        with open(self.datapath+'EvsDI_'+di+name+('_range%d' % rate_range)+'_latest.pkl','rb') as f:
            res = pickle.load(f)

        if rate_range>100:
            figsize = (1.5,1.3)
            ylim = (-70,0)
            yticks =  [-70,-10]
            rect = [0.25,0.3,0.65,0.65]
        else:
            figsize = (1.3,1.3)
            ylim = (-70,-10)
            yticks =  [-70,-20]
            rect = [0.3,0.3,0.65,0.65]

        fig=MyFigure(figsize=figsize)
        pl = fig.addplot(rect=rect)
        pl.plot(res['pre_rates'],res['vDend_mean'][0],color=colorMatrix[1],label='0')
        print res['vDend_mean'][0]
        if bothinh:
            pl.plot(res['pre_rates'],res['vDend_mean'][30],color=colorMatrix[5],label='30')
        #leg=legend(title='Inhibition (Hz)',loc=2, bbox_to_anchor=[-0.05, 1.05],frameon=False)
        pl.ax.set_ylim(ylim)
        pl.ax.set_yticks(yticks)
        xticks = [0,int(rate_range/2),rate_range]
        pl.ax.set_xticks(xticks)
        #xlabel('Rate of %d ' % p['num_inputeach']+name+' inputs (Hz)' )
        xlabel('Rate (Hz)')
        if name is 'NMDA':
            ylabel('$\overline{V}_D$ (mV)',labelpad=-5)
        else:
            pl.ax.set_yticklabels(['',''])
        if bothinh:
            fig.save(self.figpath+'EvsDI_'+name_append)
        else:
            fig.save(self.figpath+'EvsDI_preinh_'+name_append)

        if name is 'NMDA':
            inh_list = res['vDend_mean'].keys()
            inh_list = sort(inh_list)
            fig=MyFigure(figsize=(2.2,1.6))
            pl = fig.addplot(rect=[0.25,0.25,0.7,0.7])
            for i in xrange(len(inh_list)):
                color = colorMatrix[i+1,:]
                pl.plot(res['pre_rates'],res['vDend_mean'][inh_list[i]],
                        color=color,label='%d' % (inh_list[i]*k))

            leg=legend(title='Inhibition (Hz)',loc=2, bbox_to_anchor=[0.02, 1.05],frameon=False)
            pl.ax.set_ylim((-70,-10))
            pl.ax.set_yticks([-70,-20])
            pl.ax.set_xticks(xticks)
            #xlabel('Rate of %d ' % p['num_inputeach']+name+' inputs (Hz)' )
            xlabel('Excitatory input rate (Hz)')
            ylabel('Mean dendritic voltage (mV)',labelpad=5)
            fig.save(self.figpath+'EvsDI_full_'+name_append)


    def run_probabilisticNMDAplateau(self,denseinh = False):
        '''
        Plot voltage traces, showing NMDA spike are probabilistic
        '''
        rndSeed = 4
        pyrand.seed(324823+rndSeed)
        numpy.random.seed(324823+rndSeed)
        p = dict()
        p['num_DendEach']=1
        p['dt'] = 0.2*ms
        p['record_dt'] = 4.0*ms
        p['inh_base'] = 5*Hz
        p['num_input'] = 15
        vDend_list = list()
        gTotGABA_list = list()
        sNMDA_list = list()
        times_list = list()
        pre_rate_list = [50,45,40,35,30]

        di = ''
        if denseinh:
            di = 'denseinh'
            k = 10
            p['gGABA'] = p['gGABA']/k
            p['inh_base'] = p['inh_base']*k
            #pre_rate_list = [60,55,50,45,40]

        for pre_rate in pre_rate_list:
            model = MCM.Model(paramsfile=self.paramsfile,
                             eqsfile=self.eqsfile,outsidePara=p)
            model.make_model_dendrite_only(num_soma = 1,clamped_somavolt=-60*mV,condition='invivo')
            model.make_dend_bkginh(inh_rates=p['inh_base'])
            model.single_pathway_gating_experiment([pre_rate], num_input=p['num_input'],w_input = 1,record_EI=True)
            model.make_network()
            model.reinit()
            net = Network(model)
            net.run(2.0*second,report='text')

            mon = model.monitor

            times = mon['MvDend'].times/ms
            plotstart = -1
            vDend = mon['MvDend'][0][times>plotstart]/mV
            gTotGABA = mon['MgTotGABA'][0][times>plotstart]/nS
            sNMDA = mon['MsNMDA'][:,times>plotstart].mean(axis=0)
            times = times[times>plotstart]-plotstart
            vDend_list.append(vDend)
            gTotGABA_list.append(gTotGABA)
            sNMDA_list.append(sNMDA)
            times_list.append(times)

        res = dict()
        res['pre_rate_list'] = pre_rate_list
        res['times_list'] = times_list
        res['vDend_list'] = vDend_list
        res['gTotGABA_list'] = gTotGABA_list
        res['sNMDA_list'] = sNMDA_list
        res['di'] = di
        with open(self.datapath+'probabilisticNMDAplateau_latest.pkl','wb') as f:
            pickle.dump(res,f)

    def plot_probabilisticNMDAplateau(self,subject='vDend'):
        with open(self.datapath+'probabilisticNMDAplateau_latest.pkl','rb') as f:
            res = pickle.load(f)
        Np = len(res['pre_rate_list'])
        if subject is 'dendv':
            ylim_range = [-70,-10]
            ylen = 50
        elif subject is 'gTotGABA':
            ylim_range = [0,8]
            ylen = 5
        elif subject is 'sNMDA':
            ylim_range = [0,1]
            ylen = 0.5
        else:
            IOError('Unknown subject')

        fig = figure(1,(1.5,1.5))
        for i in xrange(Np):
            ax = fig.add_axes((0.2,0.9-0.7/Np*(i+1),0.6,0.7/Np))
            ax.plot(res['times_list'][i],res[subject+'_list'][i],color='black',linewidth=1)
            axis('off')
            ylim(ylim_range)
            xlim([0,res['times_list'][i][-1]])

        xlen = 500
        #ylen = 50
        xbounds = ax.get_xbound()
        ybounds = ax.get_ybound()
        lw = 1.5
        xpt2 = 1.1
        xpt1 = xpt2-xlen/(xbounds[1]-xbounds[0])
        ypt1 = -0.4
        ypt2 = ypt1+ylen/(ybounds[1]-ybounds[0])
        ax.axhline(y=ypt1*(ybounds[1]-ybounds[0])+ybounds[0],xmin=xpt1,xmax=xpt2,linewidth=lw,color='k',clip_on=False,solid_capstyle='butt')
        ax.axvline(x=xpt2*(xbounds[1]-xbounds[0])+xbounds[0],ymin=ypt1,ymax=ypt2,linewidth=lw,color='k',clip_on=False,solid_capstyle='butt')

        fig.savefig(self.figpath+'probabilisticNMDAplateau_'+subject+res['di']+self.version+'.pdf',bbox_inches='tight', pad_inches = 0)

    def plot_voltagedistribution(self,denseinh = False):
        '''
        Plot voltage distribution
        '''
        p = dict()
        p['num_DendEach']=1
        p['dt'] = 0.2*ms
        p['num_Soma'] = 40
        p['pre_rate'] = 40*Hz
        p['inh_base'] = 5*Hz

        if denseinh:
            p['gGABA'] = p['gGABA']/10
            p['inh_base'] = p['inh_base']*10
            self.version = 'denseinh_' + self.version


        pre_rates = ones(p['num_Soma'])*p['pre_rate']

        model = MCM.Model(paramsfile=self.paramsfile,eqsfile=self.eqsfile,outsidePara=p)
        model.make_model_dendrite_only(num_soma = p['num_Soma'],clamped_somavolt=-60*mV,condition='invivo')
        model.make_dend_bkginh(inh_rates=p['inh_base'])
        model.single_pathway_gating_experiment(pre_rates, num_input=15,w_input = 1)
        model.make_network()
        model.reinit()
        net = Network(model)
        net.run(2.5*second,report='text')

        mon = model.monitor

        times = mon['MvDend'].times/ms
        plotstart = 500
        dendv = mon['MvDend'].values[:,times>plotstart]/mV
        dendv = dendv[range(0,p['num_DendEach']*p['num_Soma'],p['num_DendEach']),:]

        res = dict()
        res['params'] = p
        res['dendv'] = dendv


        with open(self.datapath+'voltage_distribution_'+self.version,'wb') as f:
            pickle.dump(res,f)

        fig=MyFigure(figsize=(1.5,1.5))
        pl = fig.addplot(rect=[0.05,0.3,0.9,0.65])
        #pl.ax.set_frame_on(False)
        pl.ax.spines['left'].set_visible(False)
        pl.ax.get_yaxis().set_visible(False)
        n, bins, patches = hist(dendv.flatten(), 50, normed=1, histtype='stepfilled')
        color_hist = array([99,99,99])/255
        setp(patches, 'facecolor', color_hist,'edgecolor',color_hist)
        xlim([-70,-10])
        xticks([-60,-50,-40,-30,-20],['-60','','-40','','-20'])
        xlabel('$V_D$ (mV)')
        fig.save(self.figpath+'VoltageDistribution_'+self.version)


    def plot_NMDAPlateauMechanism(self):

        p = MCM.read_params(self.paramsfile)

        # NMDA channel voltage-glutamate input relationship
        betaPlot = linspace(0,10,100)
        def vSteadyFunc(p,v,beta):
            gL = 1
            gNMDA = beta*gL
            #gGABA = 1
            iLeak = -gL*(v-p['vRest_pyr'])
            iNMDA = -gNMDA*(v-p['vE_pyr'])/(1+exp(-(v-p['vHalfNMDA'])/p['vSpreadNMDA']))
            iGABA = -gGABA*(v-p['vI_pyr'])
            iTot = iLeak + iNMDA + iGABA
            return iTot

        gGABA=0
        figure()
        vSteadyPlot_low = list()
        vSteadyPlot_high = list()

        for beta in betaPlot:
            sol = sp.optimize.newton(lambda v: vSteadyFunc(p,v,beta),-70*mvolt,maxiter=1000)
            vSteadyPlot_low.append(sol)
            sol = sp.optimize.newton(lambda v: vSteadyFunc(p,v,beta),0*mvolt,maxiter=1000)
            vSteadyPlot_high.append(sol)

        theta = betaPlot[array(vSteadyPlot_high)/mvolt<-40][-1]
        fig = MyFigure(figsize=(1.5,1.3))
        pl=fig.addplot([0.35,0.3,0.6,0.65])
        pl.plot([theta]*2,[-70,0],'--',color=array([189,189,189])/255)
        pl.plot(betaPlot,array(vSteadyPlot_low)/mvolt, color='black')
        pl.plot(betaPlot,array(vSteadyPlot_high)/mvolt, color='black')
        pl.ax.set_yticks([-70,-10])
        pl.ax.set_xticks([0,theta,10])
        pl.ax.set_xticklabels(['0',r'$\theta_{\mathrm{NMDA}}$','10'])
        xlabel('$g_{\mathrm{NMDA}}/(g_L+g_{\mathrm{GABA}})$')
        ylabel('$V_{ss}$ (mV)',labelpad=-5)
        fig.save(self.figpath+'plot_NMDAPlateauMechanism'+self.version)

        vplot = linspace(-70,-15,100)*mV
        fMg = 1/(1+exp(-(vplot-p['vHalfNMDA'])/p['vSpreadNMDA']))
        fig = MyFigure(figsize=(1.5,1.3))
        pl=fig.addplot([0.35,0.3,0.6,0.65])
        pl.plot(vplot/mV,fMg, color='black')
        ylabel('$f_{\mathrm{Mg}}(V)$',labelpad=-5)
        xlabel('$V_D$ (mV)')
        pl.ax.set_xticks([-70,-10])
        pl.ax.set_yticks([0,0.5])
        #fig.save(self.figpath+'plot_MgVoltageDependency'+self.version)


    def plot_sNMDAmean(self):
        p = MCM.read_params(self.paramsfile)

        def meansNMDA(rate,p):
            x_mean = rate*p['tauNMDARise']
            temp = p['alphaNMDA']*p['tauNMDADecay']*x_mean
            s_mean = temp/(1+temp)
            return s_mean

        theta = 0.645 # make threshold around 30 Hz
        #theta = 0.6784
        gGABA = 0.6
        # NMDA channel voltage-glutamate input relationship
        fig = MyFigure(figsize=(1.9,1.5))
        pl=fig.addplot([0.4,0.3,0.5,0.65])
        rates = linspace(0,300,100)
        meanNMDAplot=meansNMDA(rates,p)
        meanNMDAplot2 = meanNMDAplot/(1+gGABA)
        pl.plot([0,300],[theta]*2,'--',color=array([189,189,189])/255)
        pl.plot(rates,meanNMDAplot,color=colorMatrix[1],label='0')
        pl.plot(rates,meanNMDAplot2,color=colorMatrix[5],label='%0.1f' % gGABA)

        xlabel('NMDA input rate (Hz)')
        ylabel('$g_{\mathrm{NMDA}}/(g_L+g_{\mathrm{GABA}})$')
        pl.ax.set_yticks([0,theta])
        pl.ax.set_yticklabels(['0',r'$\theta_{\mathrm{NMDA}}$'])
        pl.ax.set_xticks([0,150,300])
        leg=legend(title='$g_{\mathrm{GABA}}/g_L$',loc=4,bbox_to_anchor=[1.2, -0.05])
        leg.draw_frame(False)
        fig.save(self.figpath+'sNMDAmean'+self.version)


    def run_DendVvsgErI(self):
        '''
        Get the mean dendritic voltage as a function of excitatory input weight and
        inhibitory rate

        Used for fitting the rate model of pyramidal cells
        :return:
        '''
        p = dict()
        #p['dt'] = 0.05*ms
        p['dt'] = 0.2*ms

        p['record_dt'] = 0.5*ms
        p['num_DendEach'] = 1
        p['weights'] = arange(0.1,2.05,0.1)
        p['n_weight'] = len(p['weights'])
        p['n_rep'] = 10

        num_soma = p['n_rep']*p['n_weight']
        p['num_input'] = 15
        p['post_rate'] = 10*Hz
        p['pre_rate'] = 30*Hz

        dend_inh_rates = array([0,10,20,30,40,50,60,70,80,90,100])*Hz

        res = dict()
        res['dend_inh_rates'] = dend_inh_rates
        res['params'] = p
        res['weights'] = p['weights']
        vdend_mean_list = list()
        for dend_inh_rate in dend_inh_rates:
            model = MCM.Model(paramsfile=self.paramsfile, eqsfile=self.eqsfile,outsidePara=p)
            model.make_model(num_soma = num_soma, condition = 'invivo')
            model.weight_experiment(p['weights'], p['num_input'],
                                    p['pre_rate'], dend_inh_rate, post_rate=p['post_rate'])
            model.make_network()
            model.reinit()
            net = Network(model)
            net.run(20*second,report='text')

            mon = model.monitor

            tplot = mon['MvDend'].times

            vDend = mon['MvDend'].values[:,tplot>500*ms]
            vDend_mean = vDend.mean(axis=1) # average across time
            vdend_mean = vDend_mean.reshape((p['n_weight'],p['n_rep'])).mean(axis=1) # average across repetitions
            vdend_mean_list.append(vdend_mean)


        res['vdend_mean_list'] = vdend_mean_list
        with open(self.datapath+'DendVvsgErI'+self.version,'wb') as f:
            pickle.dump(res,f)


    def plot_DendVvsgErI(self):
        with open(self.datapath+'DendVvsgErI'+self.version,'rb') as f:
            res = pickle.load(f)
        fig=MyFigure(figsize=(4,3))
        pl = fig.addplot(rect=[0.2,0.15,0.7,0.8])
        i = 0
        for vdend_mean in res['vdend_mean_list']:
            pl.plot(res['params']['weights'],vdend_mean/mV,
                          label='%d' % res['dend_inh_rates'][i])
            xlabel('Weight')
            ylabel('Mean Dendritic Voltage (mV)')
            i += 1
        pl.ax.set_xlim((0,2))
        pl.ax.set_xticks([0,1,2])
        pl.ax.set_ylim((-70,-10))
        pl.ax.set_yticks([-70,-50,-30,-10])
        leg=legend(title='Inhibition (Hz)',loc=2)
        leg.draw_frame(False)
        fig.save(self.figpath+'DendVvsgErI'+self.version)


    def run_RatevsDendV(self):
        '''
        Get the pyramidal cell firing rate as a functio of the mean dendritic voltage
        Used for fitting the rate model of pyramidal cells
        :return:
        '''
        p = dict()
        p['dt'] = 0.2*ms
        p['num_DendEach']=10

        num_soma = 100

        runtime = 20*second

        clamped_dendvolt_list = linspace(-70,-40,11)*mV
        #clamped_dendvolt_list = array([-70,-60,-50])*mV
        fr_list = list()
        for clamped_dendvolt in clamped_dendvolt_list:
            model = MCM.Model(paramsfile=self.paramsfile, eqsfile=self.eqsfile,outsidePara=p)
            model.make_model(num_soma = num_soma,
                             condition = 'invivo',clamped_dendvolt=clamped_dendvolt)
            model.make_invivo_bkginput()
            model.make_network()
            model.reinit()

            net = Network(model)
            net.run(runtime,report='text')

            fr = model.monitor['MSpike'].nspikes/runtime/num_soma
            print 'Firing Rate %0.2f Hz' % fr
            fr_list.append(fr)

        plot(70+clamped_dendvolt_list/mV,fr_list)
        xlabel('Mean Dendrite Voltage (mV)')
        ylabel('Somatic Firing Rate (Hz)')

        result = dict()
        result['clamped_dendvolt_list'] = clamped_dendvolt_list/mV
        result['fr_list'] = fr_list
        result['params'] = p

        with open(self.datapath+'RatevsDendV'+self.version,'wb') as f:
            pickle.dump(result,f)

    def run_RatevsI(self):
        '''
        Get the pyramidal cell firing rate as a functio of the mean dendritic voltage
        Used for fitting the rate model of pyramidal cells
        :return:
        '''
        p = dict()
        p['dt'] = 0.2*ms

        num_soma = 100

        runtime = 20*second

        clamped_current_list = linspace(-200,40,21)*pamp
        fr_list = list()

        for clamped_current in clamped_current_list:
            model = MCM.Model(paramsfile=self.paramsfile, eqsfile=self.eqsfile,outsidePara=p)
            model.make_model_soma_only(num_soma = num_soma,clamped_current=clamped_current)
            model.make_invivo_bkginput()
            model.make_network()
            model.reinit()

            net = Network(model)
            net.run(runtime,report='text')

            fr = model.monitor['MSpike'].nspikes/runtime/num_soma
            print 'Firing Rate %0.2f Hz' % fr
            fr_list.append(fr)

        plot(clamped_current_list/pamp,fr_list)
        xlabel('Mean Dendrite Voltage (mV)')
        ylabel('Somatic Firing Rate (Hz)')

        result = dict()
        result['clamped_current_list'] = clamped_current_list/pamp
        result['fr_list'] = fr_list
        result['params'] = p

        with open(self.datapath+'RatevsI'+self.version,'wb') as f:
            pickle.dump(result,f)

    def run_activate_GABA_experiment(self):
        # Measure Soma-recorded IPSCs when activating GABAergic-synapse
        p = dict()

        p['num_soma'] = 1
        p['runtime'] = 0.15*second
        p['dt'] = 0.01*ms
        p['record_dt'] = p['dt']
        p['gGABA'] = 5.0*nS
        p['gNMDA'] = 2.5*nS
        p['tauGABA'] = 20*ms
        # Here the number of dendrites is conveniently set to 1, since
        # the somatic voltage is clamped at -60mV
        p['num_DendEach'] = 10
        iCurrentSoma_list = list()
        for spiketime in [50*ms,1000*ms]:
            model = MCM.Model(paramsfile=self.paramsfile, eqsfile=self.eqsfile,outsidePara=p)
            model.make_model_soma_voltageclamped(num_soma = p['num_soma'],clamped_somavolt=+10*mV)
            model.activate_GABA_experiment(spiketime=spiketime)
            model.make_network()
            model.reinit()
            net = Network(model)
            net.run(p['runtime'],report='text')

            mon = model.monitor

            plottime = mon['MvDend'].times
            vDend = mon['MvDend'].values
            iL = mon['MiL'].values
            iSyn = mon['MiSyn'].values
            iCoupleDend = mon['MiCoupleDend'].values
            iCurrentSoma = -(iL+iSyn+iCoupleDend)
            iCurrentSoma_list.append(iCurrentSoma[0,:]/pamp)

            #plot(plottime,vDend[0,:])

        IPSC = iCurrentSoma_list[0]-iCurrentSoma_list[1]
        plot(plottime[plottime>10*ms],IPSC[plottime>10*ms])
        print 'uIPSQ of dendritic GABA synapse is %0.3f pC' % (IPSC.sum()*p['record_dt'])
        #os.system('say "your program has finished"')


