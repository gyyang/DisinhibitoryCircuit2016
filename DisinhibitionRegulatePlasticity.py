'''2014-12-10
Disinhibition regulates plasticity
'''

from __future__ import division
import time, datetime
import pickle
import inspect
import numpy as np
import numpy.random
import scipy as sp
import scipy.optimize
import scipy.signal
import matplotlib.pyplot as plt
import brian_no_units
from brian import *
import MultiCompartmentalModel as MCM
from figtools import MyFigure
colorMatrix = np.array([[127,205,187],
                            [65,182,196],
                            [29,145,192],
                            [34,94,168],
                            [37,52,148],
                            [8,29,88]])/255

colorMatrix = np.array([[65,182,196],
                        [34,94,168],
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

    def invivo_learningparams(self,q):
        # in vivo change (see Higgins, Graupner, Brunel 2014)
        q['NMDA_scaling'] = q['NMDA_scaling']*0.75
        q['bAP_scaling'] = q['bAP_scaling']*0.2
        return q

    def get_weight_change(self,wpre,NMDACaTrace,bAPCaTrace,p):
        thetaD = 1
        thetaP = 1.3
        CaTrace = NMDACaTrace*p['NMDA_scaling'] + bAPCaTrace*p['bAP_scaling']
        '''
        # Original implementation to get the amount of time above threshold
        # However, this results in non-smooth objective function, making the optimization
        # much harder. In the new method, I smoothened the objective function.
        AlphaP = sum(CaTrace>thetaP)*p['dt']*p['repeat_times']
        AlphaD = sum(CaTrace>thetaD)*p['dt']*p['repeat_times']
        '''

        CaTraceSorted = sort(CaTrace)
        AlphaP = MCM.crossthresholdtime(CaTraceSorted,thetaP)*p['dt']*p['repeat_times']
        AlphaD = MCM.crossthresholdtime(CaTraceSorted,thetaD)*p['dt']*p['repeat_times']

        wpost = MCM.SynapseChange(wpre,AlphaP,AlphaD,p)
        return wpost

    def get_change_list_goal(self,experiment='NevianFig3D'):
        if experiment is 'NevianFig3D':
            change_list_goal = np.array([1.09,2.0,2.3,0.71,0.68,0.52])
        elif experiment is 'NevianFig3B':
            change_list_goal = np.array([0.97,1.04,1.95,2.01,0.80,0.72,0.68])
        elif experiment is 'PrePostAlone':
            change_list_goal = np.array([1,1])
        elif experiment is 'NevianFig2':
            change_list_goal = np.array([1.0,0.68,0.98,1.42,2.01,0.92])
        elif experiment is 'NevianFig2_partial':
            change_list_goal = np.array([1,1,0.85,0.68,0.73,0.98,1.2,1.42,1.7,2.01,1.5,0.92,1])
        else:
            IOError('Unknown experiment')
        return change_list_goal

    def get_error_bar(self,experiment='NevianFig3D'):
        if experiment is 'NevianFig3D':
            error_bar = np.array([ 0.2700157 ,  0.2197646 ,  0.47880691,  0.14128729,
                                0.0543455,  0.1195643])


        elif experiment is 'NevianFig3B':
            error_bar = np.array([ 0.08844621,  0.08556757,  0.31101381,  0.22108999,
                                0.07560724,  0.11840379,  0.05420897])

        elif experiment is 'NevianFig2':
            error_bar = np.array([0.089,0.052,0.123,0.287,0.219,0.113])
        else:
            IOError('Unknown experiment')
        return error_bar

    def compareWithNevian_CaTrace(self,para=None,remakeFig = True,experiment = 'NevianFig5',nI=40):
        if nI == 40:
            with open(self.datapath+'resultlist_'+experiment,'rb') as f:
                result_list = pickle.load(f)
        else:
            with open(self.datapath+'resultlist_'+experiment+'_nI%d' % nI,'rb') as f:
                result_list = pickle.load(f)

        if para is None:
            with open(self.datapath+'learning_para_2014-11-10_0','rb') as f:
                p = pickle.load(f)
        else:
            p = para

        N = len(result_list)
        i = 0

        fig=plt.figure(figsize=(3.0,0.7))
        for result in result_list:
            paradigm = result['paradigm']

            plottime = result['plottime']
            dt = plottime[1]-plottime[0]


            # Ca kernel
            kernel = np.exp(-plottime/p['tau_Ca_NMDA'])

            temp = p['tau_Ca_bAP_rise']/p['tau_Ca_bAP_decay']
            norm_factor = 1/(1-temp)*temp**(temp/(temp-1))

            bAPCaTrace = np.zeros(len(plottime))
            for spike_time in result['spike_times']:
                bAPCaTrace += (plottime>spike_time)*np.exp(-(plottime-spike_time)/p['tau_Ca_bAP_decay'])
                bAPCaTrace -= (plottime>spike_time)*np.exp(-(plottime-spike_time)/p['tau_Ca_bAP_rise'])
            bAPCaTrace *= norm_factor

            NMDACaTrace = scipy.signal.fftconvolve(result['iNMDA'],kernel)
            NMDACaTrace = NMDACaTrace[:len(plottime)]/p['tau_Ca_NMDA']*dt/pamp

            CaTrace = NMDACaTrace*p['NMDA_scaling'] + bAPCaTrace*p['bAP_scaling']


            tstart = 60*ms
            if paradigm['postpredelay']<0:
                post_spike_time_list = np.array([tstart+j*paradigm['spike_int'] for j in xrange(paradigm['spike_num'])])
                pre_EPSP_time = tstart - paradigm['postpredelay']
            else:
                pre_EPSP_time = tstart
                post_spike_time_list = np.array([tstart+paradigm['postpredelay']+j*paradigm['spike_int']\
                                              for j in xrange(paradigm['spike_num'])])

            ax = fig.add_axes([0.1+(0.8/N)*i,0.8,(0.8/N)*0.8,0.05])
            for post_spike_time in post_spike_time_list:
                ax.plot(np.array([post_spike_time]*2)/ms-30,[0,1],color='black')
            ax.set_ylim([0,1])
            ax.set_xlim((0,150))
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_frame_on(False)

            ax = fig.add_axes([0.1+(0.8/N)*i,0.9,(0.8/N)*0.8,0.05])
            ax.plot(np.array([pre_EPSP_time]*2)/ms-30,[0,1],color='black')
            ax.set_ylim([0,1])
            ax.set_xlim((0,150))
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_frame_on(False)



            ax = fig.add_axes([0.1+(0.8/N)*i,0.25,(0.8/N)*0.8,0.45])

            #locator_params(axis='y',nbins=3)
            #locator_params(axis='x',nbins=3)
            ax.set_yticklabels([])
            ax.set_yticks([0,4])
            ax.set_xticks([0,150])
            ax.plot([0,150],[1,1],'--',color=np.array([102,194,165])/255)
            ax.plot([0,150],[p['thetaP'],p['thetaP']],'--',color=np.array([252,141,98])/255)
            ax.plot(result['plottime']/ms-30,CaTrace,color='black')
            ax.set_ylim([0,5])
            ax.set_xlim((0,150))
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_frame_on(False)

            if i == len(result_list)-1:
                ax.text(160, 1, r'$\theta_d$',verticalalignment='center')
                ax.text(160, p['thetaP'], r'$\theta_p$',verticalalignment='center')

            if i == 0:
                xlen = 50
                ylen = 1
                xbounds = ax.get_xbound()
                ybounds = ax.get_ybound()
                lw = 1.5
                xpt1 = -0.2
                xpt2 = xpt1+xlen/(xbounds[1]-xbounds[0])
                ypt1 = -0.2
                ypt2 = ypt1+ylen/(ybounds[1]-ybounds[0])
                ax.axhline(y=ypt1*(ybounds[1]-ybounds[0])+ybounds[0],xmin=xpt1,xmax=xpt2,linewidth=lw,color='k',clip_on=False,solid_capstyle='butt')
                ax.axvline(x=xpt1*(xbounds[1]-xbounds[0])+xbounds[0],ymin=ypt1,ymax=ypt2,linewidth=lw,color='k',clip_on=False,solid_capstyle='butt')

            i+=1

        if remakeFig:
            plt.savefig(self.figpath+'compare_with'+experiment+self.version+'.pdf',format='pdf')

    def CaTrace_nonlin(self):
        '''
        calculate the nonlinearity factor of CaTrace
        res_post and res_pre are the result_list if pre or post are presented alone
        CaTrace is the CaTrace, CaTraceLinSum is the linear summation of pre and post alone
        '''
        with open(self.datapath+'resultlist_NevianFig2','rb') as f:
            resultlist = pickle.load(f)
        with open(self.datapath+'resultlist_PrePostAlone','rb') as f:
            resultlist_prepostalone = pickle.load(f)
        with open(self.datapath+'learning_para_2014-10-13_3','rb') as f:
            p = pickle.load(f)

        tau_Ca_rise = 2*ms
        tau_Ca = 12*ms
        tau_Ca_NMDA = 30*ms


        CaTrace_list = list()
        for r in resultlist_prepostalone+resultlist:
            plottime = r['plottime']
            spike_times = r['spike_times']
            iNMDA = r['iNMDA']
            dt = plottime[1]-plottime[0]
            kernel = np.exp(-plottime/tau_Ca_NMDA)
            temp = tau_Ca_rise/tau_Ca
            norm_factor = 1/(1-temp)*temp**(temp/(temp-1))

            bAPCaTrace = np.zeros(len(plottime))
            for spike_time in spike_times:
                bAPCaTrace += (plottime>spike_time)*np.exp(-(plottime-spike_time)/tau_Ca)
                bAPCaTrace -= (plottime>spike_time)*np.exp(-(plottime-spike_time)/tau_Ca_rise)
            bAPCaTrace *= norm_factor

            NMDACaTrace = scipy.signal.fftconvolve(iNMDA,kernel)
            NMDACaTrace = NMDACaTrace[:len(plottime)]/tau_Ca_NMDA*dt/pamp

            CaTrace = NMDACaTrace*p['NMDA_scaling'] + bAPCaTrace*p['bAP_scaling']
            CaTrace_list.append(CaTrace)

        postpredelay_list = list()
        nonlinfactor_list = list()
        maxCa_list = list()
        for ind in xrange(len(resultlist)):
            res = resultlist[ind]
            postpredelay = res['paradigm']['postpredelay']
            postpredelay_list.append(postpredelay)
            if postpredelay<0:
                # shift res_pre CaTrace
                CaTrace_pre = CaTrace_list[1]
                shiftind = int(-postpredelay/dt)
                CaTrace_pre = concatenate((np.zeros(shiftind),CaTrace_pre[:-shiftind]))
                CaTraceLinSum = CaTrace_list[0] + CaTrace_pre
            else:
                # shift res_post CaTrace
                CaTrace_post = CaTrace_list[0]
                shiftind = int(postpredelay/dt)
                CaTrace_post = concatenate((np.zeros(shiftind),CaTrace_post[:-shiftind]))
                CaTraceLinSum = CaTrace_list[1] + CaTrace_post
            CaTrace = CaTrace_list[ind+2]

            nonlinfactor = max(CaTrace)/max(CaTraceLinSum)
            nonlinfactor_list.append(nonlinfactor)

            maxCa_list.append(max(CaTrace))
        postpredelay_list = np.array(postpredelay_list)/ms
        plot(postpredelay_list,nonlinfactor_list)
        plt.ylabel('non-linearity factor')
        plt.xlabel('post-pre delay (ms)')

        '''
        plot(CaTrace)
        plot(CaTraceLinSum)
        plt.legend(['Trace','Linear Sum'])
        '''

    def plot_model_result(self,pl, para, result_list, experiment, plot_data=False):
        '''

        :param pl: plot handle
        :param para:
        :param result_list:
        :param experiment:
        :param plot_data:
        :return:
        '''

        datacolor1 = np.array([107,174,214])/255
        datacolor2 = np.array([251,106,74])/255
        datacolor3 = np.array([150,150,150])/255
        modelcolor1 = np.array([8,81,156])/255
        modelcolor2 = np.array([165,15,21])/255
        modelcolor3 = np.array([37,37,37])/255
        elw = 1.5
        mks = 4
        if experiment is 'NevianFig3B':
            sqe, change_list, wpost_list, para = get_model_sqe(para, result_list,
                                                     rich_return=True, recalc_Ca=True)

            change_list_goal = get_change_list_goal(experiment)
            error_bar = get_error_bar('NevianFig3B')
            ap_numbers = list()
            for res in result_list:
                ap_numbers.append(res['paradigm']['spike_num'])

            pl.plot([-0.2,3.2],[1,1],color=np.array([189,189,189])/255)
            errorbar(ap_numbers[0],change_list_goal[0],yerr=error_bar[0],fmt='D',
                     markerfacecolor=datacolor3,markeredgecolor=datacolor3,
                     markersize=mks,ecolor=datacolor3,capsize=1, elinewidth=elw)
            pl.plot(ap_numbers[0],change_list[0],'d',color=modelcolor3,
                    markeredgecolor=modelcolor3,markersize=mks)

            errorbar(ap_numbers[1:4],change_list_goal[1:4],yerr=error_bar[1:4],fmt='o',
                     markeredgecolor=datacolor1,markerfacecolor=datacolor1,
                     markersize=mks,ecolor=datacolor1,capsize=1, elinewidth=elw)
            pl.plot(ap_numbers[1:4],change_list[1:4],color=modelcolor1)

            errorbar(ap_numbers[4:],change_list_goal[4:],yerr=error_bar[4:],fmt='s',
                     markeredgecolor=datacolor2,markerfacecolor=datacolor2,
                     markersize=mks,ecolor=datacolor2,capsize=1, elinewidth=elw)
            pl.plot(ap_numbers[4:],change_list[4:],color=modelcolor2)

            pl.ax.set_xticks([0,1,2,3])
            pl.ax.set_yticks([0.5,1.0,1.5,2.0,2.5])

            plt.xlabel('Number of spikes')
            plt.ylabel('Weight change')
            plt.xlim([-0.2,3.2])
            plt.ylim([0.25,2.75])

        elif experiment in ['NevianFig3D','NevianFig3D_full']:

            pl.plot([0,110],[1,1],color=np.array([189,189,189])/255)
            change_list_goal = get_change_list_goal('NevianFig3D')
            error_bar = get_error_bar('NevianFig3D')

            sqe, change_list, wpost_list, para = get_model_sqe(para, result_list,
                                                     rich_return=True, recalc_Ca=True)

            frequencies = list()
            for res in result_list:
                frequencies.append(1/res['paradigm']['spike_int'])
            N = len(result_list)//2

            errorbar([20,50,100],change_list_goal[0:3],yerr=error_bar[0:3],fmt='o',
                     markeredgecolor=datacolor1,markerfacecolor=datacolor1,markersize=mks,
                     ecolor=datacolor1,capsize=1, elinewidth=elw)
            pl.plot(frequencies[0:N],change_list[0:N],color=modelcolor1)

            errorbar([20,50,100],change_list_goal[3:],yerr=error_bar[3:],fmt='s',
                     markeredgecolor=datacolor2,markerfacecolor=datacolor2,markersize=mks,
                     ecolor=datacolor2,capsize=1, elinewidth=elw)
            pl.plot(frequencies[N:],change_list[N:],color=modelcolor2)


            plt.xlabel('AP frequency (Hz)')
            plt.ylabel('Weight change')

            pl.ax.set_xticks([20,50,100])
            pl.ax.set_yticks([0.5,1.0,1.5,2.0,2.5])
            plt.xlim([0,110])
            plt.ylim([0.25,2.75])

        elif experiment in ['NevianFig2_full','NevianFig2']:
            sqe, change_list, wpost_list, para = get_model_sqe(para, result_list,
                                                     rich_return=True, recalc_Ca=True)
            delays = list()
            for res in result_list:
                delays.append(res['paradigm']['postpredelay'])
            delays = np.array(delays)/ms
            pl.plot([-100,50],[1,1],color=np.array([189,189,189])/255)
            change_list_goal = get_change_list_goal('NevianFig2')
            error_bar = get_error_bar('NevianFig2')
            errorbar(np.array([-90,-50,-30,-10,10,50]),change_list_goal,yerr=error_bar,
                     fmt='s',markerfacecolor=datacolor3,markeredgecolor=datacolor3,
                     markersize=mks,ecolor=datacolor3,capsize=1, elinewidth=elw)

            pl.plot(delays,change_list,color=modelcolor3)

            plt.xlabel('Time lag (ms)')
            plt.ylabel('Weight change')
            pl.ax.set_xticks([-100,-50,0,50])
            pl.ax.set_yticks([0.5,1.0,1.5,2.0,2.5])
            plt.xlim([-100,55])
            plt.ylim([0.25,2.75])

        elif experiment in ['prepostequal','preonly','prepostequal_inh','preonly_inh']:
            sqe, change_list, wpost_list, para = get_model_sqe(para, result_list,
                                                     rich_return=True, recalc_Ca=False)
            colorMatrix = np.array([[29,145,192],[8,29,88]])/255
            if experiment is 'prepostequal':
                xlabeltext = 'Pre=post rate (Hz)'
                color = colorMatrix[0,:]
            elif experiment is 'prepostequal_inh':
                xlabeltext = 'Pre=post rate (Hz)'
                color = colorMatrix[1,:]
            elif experiment is 'preonly':
                xlabeltext = 'Pre alone rate (Hz)'
                color = colorMatrix[0,:]
            elif experiment is 'preonly_inh':
                xlabeltext = 'Pre alone rate (Hz)'
                color = colorMatrix[1,:]
            rates = list()
            vdend_mean = list()
            for res in result_list:
                rates.append(res['paradigm']['pre_rate'])
                vdend_mean.append(res['vDend'].mean())
            pl.plot(rates,change_list,color=color)
            locator_params(axis='x',nbins=5)
            locator_params(axis='y',nbins=4)
            plt.xlabel(xlabeltext)
            plt.ylabel('Weight change')


    def run_spiking_experiments(self,experiment='NevianFig3D',restore=True):
        paradigms = list()
        nI = 40 # 40 simultaneous input is enough to bring dend close to NMDA plateau

        if experiment is 'NevianFig3D':
            paradigms.append({'spike_num':3,'spike_int':50*ms,'postpredelay':10*ms,'num_input':nI})
            paradigms.append({'spike_num':3,'spike_int':20*ms,'postpredelay':10*ms,'num_input':nI})
            paradigms.append({'spike_num':3,'spike_int':10*ms,'postpredelay':10*ms,'num_input':nI})

            paradigms.append({'spike_num':3,'spike_int':50*ms,'postpredelay':-110*ms,'num_input':nI})
            paradigms.append({'spike_num':3,'spike_int':20*ms,'postpredelay':-50*ms,'num_input':nI})
            paradigms.append({'spike_num':3,'spike_int':10*ms,'postpredelay':-30*ms,'num_input':nI})
        elif experiment is 'NevianFig3D_full':
            spike_int_plot = linspace(50,10,51)*ms
            for spike_int in spike_int_plot:
                paradigms.append({'spike_num':3,'spike_int':spike_int,'postpredelay':10*ms,'num_input':nI})

            for spike_int in spike_int_plot:
                paradigms.append({'spike_num':3,'spike_int':spike_int,'postpredelay':-2*spike_int-10*ms,'num_input':nI})
        elif experiment is 'NevianFig3B':
            paradigms.append({'spike_num':0,'spike_int':50*ms,'postpredelay':100*ms,'num_input':nI})
            paradigms.append({'spike_num':1,'spike_int':20*ms,'postpredelay':10*ms,'num_input':nI})
            paradigms.append({'spike_num':2,'spike_int':20*ms,'postpredelay':10*ms,'num_input':nI})
            paradigms.append({'spike_num':3,'spike_int':20*ms,'postpredelay':10*ms,'num_input':nI})

            paradigms.append({'spike_num':1,'spike_int':20*ms,'postpredelay':-10*ms,'num_input':nI})
            paradigms.append({'spike_num':2,'spike_int':20*ms,'postpredelay':-30*ms,'num_input':nI})
            paradigms.append({'spike_num':3,'spike_int':20*ms,'postpredelay':-50*ms,'num_input':nI})
        elif experiment is 'NevianFig5':
            paradigms.append({'spike_num':3,'spike_int':20*ms,'postpredelay':1000*ms,'num_input':nI})
            paradigms.append({'spike_num':3,'spike_int':20*ms,'postpredelay':-1000*ms,'num_input':nI})
            paradigms.append({'spike_num':3,'spike_int':20*ms,'postpredelay':-50*ms,'num_input':nI})
            paradigms.append({'spike_num':3,'spike_int':20*ms,'postpredelay':10*ms,'num_input':nI})
        elif experiment is 'PrePostAlone':
            paradigms.append({'spike_num':3,'spike_int':20*ms,'postpredelay':-10000*ms,'num_input':nI})
            paradigms.append({'spike_num':3,'spike_int':20*ms,'postpredelay':10000*ms,'num_input':nI})
        elif experiment is 'NevianFig2':
            Dt = 20*ms
            postpredelay_list = np.array([-90,-50,-30,-10,10,50])*ms
            for postpredelay in postpredelay_list:
                paradigms.append({'spike_num':3,'spike_int':Dt,'postpredelay':postpredelay,'num_input':nI})
        elif experiment is 'NevianFig2_full':
            Dt = 20*ms
            postpredelay_list = linspace(-100,50,51)*ms
            for postpredelay in postpredelay_list:
                paradigms.append({'spike_num':3,'spike_int':Dt,'postpredelay':postpredelay,'num_input':nI})
        else:
            IOError('Unknown experiment')


        runtime = 500*ms

        result_list = list()
        for i in xrange(len(paradigms)):
            model = MCM.Model(paramsfile=self.paramsfile, eqsfile=self.eqsfile)
            model.make_model()
            tstart = 60*ms
            spike_num = paradigms[i]['spike_num']
            postpredelay = paradigms[i]['postpredelay']
            spike_int = paradigms[i]['spike_int']
            num_input = paradigms[i]['num_input']
            if postpredelay<0:
                post_spike_time_list = np.array([tstart+i*spike_int for i in xrange(spike_num)])
                pre_EPSP_time = tstart - postpredelay
            else:
                pre_EPSP_time = tstart
                post_spike_time_list = np.array([tstart+postpredelay+i*spike_int for i in xrange(spike_num)])
            model.spike_train_experiment(pre_EPSP_time, post_times=post_spike_time_list, num_input=num_input)
            model.make_network()
            model.reinit()
            net = Network(model)
            net.run(runtime,report='text')
            mon = model.monitor

            result = dict()
            result['plottime'] = mon['MNMDACasyn'].times
            result['spike_times'] = mon['MSpike'][0]
            result['NMDACaTrace'] = mon['MNMDACasyn'][0]
            result['bAPCa'] = mon['MbAPCa'][0]
            result['iNMDA'] = mon['MiNMDAsyn'][0]
            result['vDend'] = mon['MvDend'][0]
            result['vSoma'] = mon['MvSoma'][0]
            result['paradigm'] = paradigms[i]
            result_list.append(result)

        if restore:
            print 'data file restored'
            with open(self.datapath+'resultlist_'+experiment,'wb') as f:
                pickle.dump(result_list,f)

        return result_list

    def get_model_sqe(self,para, result_list, experiment=None,
                      rich_return=False, recalc_Ca=True):
        '''
        para is a list of parameters
        return squared error
        '''
        #print para
        p = dict()
        plottime = result_list[0]['plottime']
        p['dt'] = plottime[1]-plottime[0]
        p['repeat_times'] = 60
        p['rhoStar'] = 0.5 # boundary of basins of attraction of two stable states


        #p['gammaD'] = 332
        #p['gammaP'] = 725
        #p['thetaP'] = 2.86

        # Parameters got from Graupner & Brunel
        p['thetaD'] = 1 # arbitrary
        p['sigma'] = 3.35 # noise level
        p['tau'] = 346.36
        # From Nevian & Sakmann
        p['tau_Ca'] = 30 # ms


        p['w0'] = 0 # strength in low state, changed to 0
        p['w1'] = 3 # strength in high state


        if (type(para) is list) or (type(para) is numpy.ndarray):
            para_name_list = ['NMDA_scaling','bAP_scaling','gammaD','gammaP','thetaP']
            j = 0
            for para_name in para_name_list:
                p[para_name] = para[j]
                j += 1
        elif type(para) is dict:
            for k in para.keys():
                p[k] = para[k]
        else:
            raise ValueError('Unknown para type')

        thetaD = p['thetaD']
        thetaP = p['thetaP']
        p['tau_Ca_bAP_rise'] = 2*ms
        p['tau_Ca_bAP_decay'] = p['tau_Ca']*ms
        p['tau_Ca_NMDA'] = p['tau_Ca']*ms
        wpre = 1


        # Ca kernel
        kernel = np.exp(-plottime/p['tau_Ca_NMDA'])

        temp = p['tau_Ca_bAP_rise']/p['tau_Ca_bAP_decay']
        norm_factor = 1/(1-temp)*temp**(temp/(temp-1))

        wpost_list = list()

        for res in result_list:
            if recalc_Ca: # recalculate Ca from spike timing and NMDA current
                # this is most appropriate for spike experiment, not for rate experiments.
                bAPCaTrace = np.zeros(len(plottime))
                for spike_time in res['spike_times']:
                    bAPCaTrace += (plottime>spike_time)*np.exp(-(plottime-spike_time)/p['tau_Ca_bAP_decay'])
                    bAPCaTrace -= (plottime>spike_time)*np.exp(-(plottime-spike_time)/p['tau_Ca_bAP_rise'])
                bAPCaTrace *= norm_factor

                NMDACaTrace = scipy.signal.fftconvolve(res['iNMDA'],kernel)
                NMDACaTrace = NMDACaTrace[:len(plottime)]/p['tau_Ca_NMDA']*p['dt']/pamp

                CaTrace = NMDACaTrace*p['NMDA_scaling'] + bAPCaTrace*p['bAP_scaling']
                CaTraceSorted = sort(CaTrace)
                AlphaP = MCM.crossthresholdtime(CaTraceSorted,thetaP)*p['dt']*p['repeat_times']
                AlphaD = MCM.crossthresholdtime(CaTraceSorted,thetaD)*p['dt']*p['repeat_times']
            else:
                NMDACasyn = res['NMDACasyn'][:,plottime>500*ms] # get values for t > 0.5s
                bAPCasyn = res['bAPCasyn'][:,plottime>500*ms] # get values for t > 0.5s

                num_syn = NMDACasyn.shape[0]

                # put together Ca trace from multiple synapses
                NMDACaTrace = NMDACasyn.flatten()
                bAPCaTrace = bAPCasyn.flatten()
                CaTrace = NMDACaTrace*p['NMDA_scaling'] + bAPCaTrace*p['bAP_scaling']
                CaTraceSorted = sort(CaTrace)
                AlphaP = MCM.crossthresholdtime(CaTraceSorted,thetaP)*p['dt']*p['repeat_times']/num_syn
                AlphaD = MCM.crossthresholdtime(CaTraceSorted,thetaD)*p['dt']*p['repeat_times']/num_syn




            wpost = MCM.SynapseChange(wpre,AlphaP,AlphaD,p)

            wpost_list.append(wpost)

        change_list = np.array(wpost_list)/wpre
        if experiment is not None:
            change_list_goal = get_change_list_goal(experiment)
            #sqe = sum((change_list_goal-change_list)**2)
            sqe = sum((change_list/change_list_goal-1)**2)
        else:
            sqe = -1 # not used


        if isnan(sqe):
            print('current para = '+' '.join('{:0.3f}'.format(k) for k in para))
            raise ValueError

        if not rich_return:
            return sqe
        else:
            return sqe, change_list, wpost_list, p

    def fit_experiment(self):
        global Nfeval
        experiment3B = 'NevianFig3B'
        with open(self.datapath+'resultlist_'+experiment3B,'rb') as f:
            result_list3B = pickle.load(f)
        experiment3D = 'NevianFig3D'
        with open(self.datapath+'resultlist_'+experiment3D,'rb') as f:
            result_list3D = pickle.load(f)
        experiment3D_full = 'NevianFig3D_full'
        with open(self.datapath+'resultlist_'+experiment3D_full,'rb') as f:
            result_list3D_full = pickle.load(f)
        experiment2 = 'NevianFig2'
        with open(self.datapath+'resultlist_'+experiment2,'rb') as f:
            result_list2 = pickle.load(f)
        experiment2_full = 'NevianFig2_full'
        with open(self.datapath+'resultlist_'+experiment2_full,'rb') as f:
            result_list2_full = pickle.load(f)


        x0 = list()
        bnds = list()
        # The following is the most up-to-date parameters fitting Nevian Fig.2

        x0.append(0.371)    # NMDA scaling parameter
        bnds.append([0.3,3])
        x0.append(0.957)    # bAP scaling
        bnds.append([0.4,1])
        x0.append(39.949)      # induced depression strength
        bnds.append([20,500])
        x0.append(177.552)  # induced potentiation strength
        bnds.append([50,500])
        #x0.append(346.833)      # synaptic learning time constant (s)
        #bnds.append([50,700])
        x0.append(2.78)      # potentiation threshold (this parameter is very important)
        bnds.append([1.2,3])
        #x0.append(20)
        #bnds.append([12,100]) # Ca time constant (ms)
        #x0.append(0.647)
        #bnds.append([0.35,70]) # noise constant sigma

        start = time.clock()
        # Fit the model to some of the data but not all
        obj_func = lambda para: \
        0*self.get_model_sqe(para, result_list3B, experiment3B) + \
        1*self.get_model_sqe(para, result_list3D, experiment3D) + \
        1*self.get_model_sqe(para, result_list2, experiment2)
        Nfeval = 1
        def callbackF(Xi):
            global Nfeval
            if Nfeval == 1:
                print 'N  f     NMDA   bAP   wpre    w1   gP      tau      tau_Ca'
            print('%2d %0.3f ' % (Nfeval,obj_func(Xi))+' '.join('{:0.3f}'.format(k) for k in Xi))

            Nfeval += 1
        res = scipy.optimize.minimize(obj_func,x0,bounds=bnds,method='SLSQP',
                                      options={'maxiter':300, 'disp':True},callback=callbackF)
        x = res.x
        print('%2d %0.3f ' % (Nfeval,obj_func(x))+' '.join('{:0.3f}'.format(k) for k in x))


        end = time.clock()
        print 'time spent %0.6f' % (end-start)


        sqe, change_list, wpost_list, para = self.get_model_sqe(x, result_list2, rich_return=True)

        fig=MyFigure(figsize=(2,2))
        pl = fig.addplot(rect=[0.2,0.15,0.7,0.8])
        self.plot_model_result(pl, para, result_list2_full, experiment2_full, plot_data=True)
        self.compareWithNevian_CaTrace(para,remakeFig = False,experiment = 'NevianFig5')

        fig=MyFigure(figsize=(1.5,2))
        pl = fig.addplot(rect=[0.2,0.15,0.7,0.8])
        self.plot_model_result(pl, para, result_list3B, experiment3B, plot_data=True)
        #compareWithNevian_CaTrace(para,remakeFig = False,experiment = experiment3B)


        fig=MyFigure(figsize=(1.5,2))
        pl = fig.addplot(rect=[0.2,0.15,0.7,0.8])
        self.plot_model_result(pl, para, result_list3D_full, experiment3D_full, plot_data=True)
        #compareWithNevian_CaTrace(para,remakeFig = False,experiment = experiment3D)


    def plot_NevianFit(self):
        with open(self.datapath+'learning_para_2014-11-10_0','rb') as f:
            para = pickle.load(f)
        experiment3B = 'NevianFig3B'
        with open(self.datapath+'resultlist_'+experiment3B,'rb') as f:
            result_list3B = pickle.load(f)
        experiment3D_full = 'NevianFig3D_full'
        with open(self.datapath+'resultlist_'+experiment3D_full,'rb') as f:
            result_list3D_full = pickle.load(f)
        experiment2_full = 'NevianFig2_full'
        with open(self.datapath+'resultlist_'+experiment2_full,'rb') as f:
            result_list2_full = pickle.load(f)

        remakeFig = True

        fig=MyFigure(figsize=(1.5,1.5))
        pl = fig.addplot(rect=[0.3,0.25,0.65,0.7])
        self.plot_model_result(pl, para, result_list2_full, experiment2_full, plot_data=True)
        #fig.save(self.figpath+'Fit'+experiment2_full+'_'+self.version)
        #compareWithNevian_CaTrace(para,remakeFig,experiment = 'NevianFig5')

        fig=MyFigure(figsize=(1.5,1.5))
        pl = fig.addplot(rect=[0.3,0.25,0.65,0.7])
        self.plot_model_result(pl, para, result_list3B, experiment3B, plot_data=True)
        #fig.save(self.figpath+'Fit'+experiment3B+'_'+self.version)
        #compareWithNevian_CaTrace(para,remakeFig,experiment = experiment3B)


        fig=MyFigure(figsize=(1.5,1.5))
        pl = fig.addplot(rect=[0.3,0.25,0.65,0.7])
        self.plot_model_result(pl, para, result_list3D_full, experiment3D_full, plot_data=True)
        fig.save(self.figpath+'Fit'+experiment3D_full+'_'+self.version)
        #compareWithNevian_CaTrace(para,remakeFig,experiment = experiment3D_full)


    def run_WeightChangevsInitWeight(self):
        p = dict()
        p['dt'] = 0.2*ms
        p['record_dt'] = 0.5*ms
        p['num_DendEach'] = 1
        p['weights'] = arange(0.1,3.05,0.1)
        p['n_weight'] = len(p['weights'])
        p['n_rep'] = 1
        num_soma = p['n_rep']*p['n_weight']
        p['num_input'] = 15
        p['post_rate'] = 20*Hz

        p['runtime'] = 5*second


        pre_rate = 50*Hz
        dend_inh_rates = np.array([5,20,35,50,65,80])*Hz

        resultlist = list()
        for dend_inh_rate in dend_inh_rates:
            model = MCM.Model(paramsfile=self.paramsfile, eqsfile=self.eqsfile,outsidePara=p)
            model.make_model_dendrite_only(num_soma = num_soma,clamped_somavolt=-60*mV,condition='invivo')
            model.weight_experiment(p['weights'], p['num_input'],
                                    pre_rate, dend_inh_rate, post_rate=p['post_rate'])
            model.make_network()
            model.reinit()
            net = Network(model)
            net.run(p['runtime'],report='text')

            mon = model.monitor

            tplot = mon['MvDend'].times

            vDend = mon['MvDend'].values[:,tplot>500*ms]
            NMDACasyn = mon['MNMDACasyn'].values[:,tplot>500*ms]
            bAPCa = mon['MbAPCa'].values[:,tplot>500*ms]

            p['pre_rate'] = pre_rate
            result = dict()
            result['params'] = p
            result['vDend'] = vDend
            result['NMDACasyn'] = NMDACasyn
            result['bAPCa'] = bAPCa
            result['dend_inh_rate'] = dend_inh_rate
            resultlist.append(result)

        with open(self.datapath+'WeightChangevsInitWeight_postr%d' % p['post_rate'],'wb') as f:
            pickle.dump(resultlist,f)

        return resultlist

    def plot_WeightChangevsInitWeight(self):
        post_rate = 10
        with open(self.datapath+'learning_para_2014-11-10_0','rb') as f:
            para = pickle.load(f)
        q = para.copy()
        with open(self.datapath+'WeightChangevsInitWeight_postr%d' % post_rate,'rb') as f:
            resultlist = pickle.load(f)
        # in vivo change (see Higgins, Graupner, Brunel 2014)
        q = self.invivo_learningparams(q)

        vDend_mean_list = list()

        fig=MyFigure(figsize=(4,3))
        pl = fig.addplot(rect=[0.2,0.15,0.7,0.8])
        pl.plot([q['w0'],q['w1']],[q['w0'],q['w1']],color='black',linestyle='--')

        i = 0
        for result in resultlist:
            p = result['params']
            vDend = result['vDend']
            NMDACasyn = result['NMDACasyn']
            bAPCa = result['bAPCa']

            vDend_mean = vDend.mean(axis=1)

            vDend_mean = vDend_mean.reshape(p['n_weight'],p['n_rep'])
            vDend_mean = vDend_mean.mean(axis=1)
            vDend_mean_list.append(vDend_mean)


            num_syn = NMDACasyn.shape[0]
            num_soma = bAPCa.shape[0]
            synsomaratio = num_syn//num_soma

            bAPCasyn = bAPCa.repeat(synsomaratio,axis=0)
            CaTrace = NMDACasyn*q['NMDA_scaling'] + bAPCasyn*q['bAP_scaling']

            w_post = np.zeros(num_syn)

            n_eachweight = num_syn//p['n_weight']
            total_time = 3000 #s
            record_dt = p['record_dt']
            for i_syn in xrange(num_syn):
                CaTraceSorted = np.sort(CaTrace[i_syn])

                AlphaP = MCM.crossthresholdtime(CaTraceSorted,q['thetaP'])*record_dt*(total_time/p['runtime'])
                AlphaD = MCM.crossthresholdtime(CaTraceSorted,q['thetaD'])*record_dt*(total_time/p['runtime'])

                wpre = p['weights'][i_syn//n_eachweight]
                wpost_syn = MCM.SynapseChange(wpre,AlphaP,AlphaD,q)
                w_post[i_syn] = wpost_syn

            w_post = w_post.reshape(p['n_weight'],n_eachweight)
            w_post = w_post.mean(axis=1)

            ax, = pl.plot(p['weights'],w_post,color=colorMatrix[i*0,:],label='%d' % result['dend_inh_rate'])
            i += 1

        plt.xlabel('Weight before learning')
        plt.ylabel('Weight after learning')
        pl.ax.set_plt.xlim((0,3))
        pl.ax.set_xticks([0,1,2,3])
        pl.ax.set_plt.ylim((0,3))
        pl.ax.set_yticks([0,1,2,3])
        leg=plt.legend(title='Inhibition (Hz)',loc=2)
        leg.draw_frame(False)
        axis('equal')
        #title('%d NMDA inputs of %d Hz' % (num_input,pre_rate))
        fig.save(self.figpath+('WeightChangevsInitWeight_%d' % post_rate)+self.version)

    def run_WeightChangevsRate(self,restore=True):
        restore = True
        p = dict()
        p['dt'] = 0.2*ms
        p['record_dt'] = 2.0*ms
        p['num_DendEach'] = 1
        p['pre_rates'] = linspace(0,50,21)*Hz
        p['n_rate'] = len(p['pre_rates'])
        p['n_rep'] = 1
        num_soma = p['n_rep']*p['n_rate']
        p['num_input'] = 15
        p['post_rate'] = 10*Hz

        dend_inh_rates = array([0,20,40])*Hz

        p['dend_inh_rates'] = dend_inh_rates

        p['runtime'] = 100*second

        self.result_list = list()
        for dend_inh_rate in dend_inh_rates:
            model = MCM.Model(paramsfile=self.paramsfile, eqsfile=self.eqsfile,outsidePara=p)
            # mimic the in vivo condition by reducing coupling weight, and
            # clamping somatic membrane potential around -60mV, but not providing direct EI input to soma
            model.make_model_dendrite_only(num_soma = num_soma,clamped_somavolt=-60*mV,condition='invivo')
            model.rate_experiment(num_input=p['num_input'], pre_rates = p['pre_rates'],
                            post_rate = p['post_rate'], dend_inh_rate=dend_inh_rate)

            model.make_network()
            model.reinit()
            net = Network(model)
            net.run(p['runtime'],report='text')
            mon = model.monitor

            tplot = mon['MvDend'].times

            p['dend_inh_rate'] = dend_inh_rate
            result = dict()
            result['params'] = p
            result['vDend'] = mon['MvDend'].values[:,tplot>500*ms]
            result['NMDACasyn'] = mon['MNMDACasyn'].values[:,tplot>500*ms]
            result['bAPCa'] = mon['MbAPCa'].values[:,tplot>500*ms]
            result['dend_inh_rate'] = dend_inh_rate
            self.result_list.append(result)

        if restore:
            with open(self.datapath+'WeightChangevsRate_postr%d' % p['post_rate'],'wb') as f:
                pickle.dump(self.result_list,f)

        #return result_list

    def calc_WeightChangevsRate(self):
        if hasattr(self,'result_list'):
            print 'Directly use the result_list just computed'
            post_rate = self.result_list[0]['params']['post_rate']
        else:
            post_rate = 10
            filename = self.datapath+'WeightChangevsRate_postr%d' % post_rate
            print 'Loading result_list from file: '+filename
            with open(filename,'rb') as f:
                self.result_list = pickle.load(f)

        with open(self.datapath+'learning_para_2014-11-10_0','rb') as f:
            para = pickle.load(f)

        q = para.copy()
        q = self.invivo_learningparams(q)

        vDend_mean_list = list()
        self.w_post_list = list()
        for result in self.result_list:
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

            w_post = np.zeros(num_syn)

            n_eachrate = num_syn//p['n_rate']
            total_time = 3000 #s
            record_dt = p['record_dt']
            for i_syn in xrange(num_syn):
                CaTraceSorted = sort(CaTrace[i_syn])

                AlphaP = MCM.crossthresholdtime(CaTraceSorted,q['thetaP'])*record_dt*(total_time/p['runtime'])
                AlphaD = MCM.crossthresholdtime(CaTraceSorted,q['thetaD'])*record_dt*(total_time/p['runtime'])

                wpre = 1
                wpost_syn = MCM.SynapseChange(wpre,AlphaP,AlphaD,q)
                w_post[i_syn] = wpost_syn

            w_post = w_post.reshape(p['n_rate'],n_eachrate)
            w_post = w_post.mean(axis=1)
            self.w_post_list.append(w_post)

        output = {'w_post_list':self.w_post_list, 'params':p}
        with open(self.datapath+'calc_WeightChangevsRate','wb') as f:
            pickle.dump(output,f)


    def plot_WeightChangevsRate(self):
        with open(self.datapath+'calc_WeightChangevsRate','rb') as f:
            output = pickle.load(f)

        p = output['params']
        w_post_list = output['w_post_list']
        dend_inh_rates = p['dend_inh_rates']


        colorMatrix = array([[65,182,196],
                            [34,94,168],
                            [8,29,88]])/255


        fig=MyFigure(figsize=(2.5,1.2))
        pl = fig.addplot(rect=[0.25,0.3,0.7,0.65])
        pl.plot([0,max(p['pre_rates'])],[1,1],color=array([189,189,189])/255)
        i = 0
        plotlist = list()
        for w_post in w_post_list:
            ax, = pl.plot(p['pre_rates'],w_post,color=colorMatrix[i,:])
            plotlist.append(ax)
            plt.xlabel('Pre-synaptic rate (Hz)')
            plt.ylabel('Weight change')
            i += 1
        pl.ax.set_xticks([0,20,40])
        pl.ax.set_plt.ylim((0,3))
        pl.ax.set_yticks([0,1,2,3])
        leg=plt.legend(plotlist,['%d' % r for r in dend_inh_rates],title='Inhibition (Hz)',
                   loc=2,bbox_to_anchor=[0.1, 1.15])
        leg.draw_frame(False)
        fig.save(self.figpath+'WeightChangevsRate'+self.version)


    