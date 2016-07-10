'''2014-11
Modeling Mante et al. 2013 task
'''

from __future__ import division
from sys import exc_clear
import time
import os
import re
import datetime
import numpy as np
import numpy.random
import bisect
import scipy as sp
import scipy.signal
import scipy.optimize
import scipy.misc
import random as pyrand
import pylab
import pickle
from figtools import MyFigure
import MultiCompartmentalModel as MCM
import WongWangModel as WW
import InterneuronalCircuit

dend_IO = MCM.dend_IO
soma_IO = MCM.soma_IO
soma_fv = MCM.soma_fv

    
motion_colors = np.array([[37,37,37],[99,99,99],[150,150,150]])/255
color_colors = np.array([[8,104,172],[67,162,202],[123,204,196]])/255
motion_colors2 = np.array([[189,0,38],[240,59,32],[253,141,60]])/255
motion_colors3 = np.array([[177,0,38],[252,78,42],[254,178,76]])/255

motion_colors4 = np.array([[179,205,227],[140,150,198],[136,86,167],[129,15,124]])/255

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


    def set_EI(self,n_som2dend=None,seed=0,no_selectivity=False):
        numpy.random.seed(seed)
        pyrand.seed(seed)

        IC = InterneuronalCircuit.Study(figpath=figpath,datapath=datapath,
                                     version=version,
                                     paramsfile=datapath+'parameters.txt',
                                     eqsfile=datapath+'equations.txt')
        self.params = IC.get_p(modeltype='Control2VIPSOM')

        if no_selectivity:
            self.params['p_exc2som'] = 1.0
            self.params['p_exc2vip'] = 1.0

        if n_som2dend is not None:
            p_som2dend = n_som2dend/self.params['n_som']
            self.params['p_som2pyr'] = 1-(1-p_som2dend)**self.params['n_dend_each']
            print 'n_som2dend = %d, p_som2pyr = %0.3f' % (n_som2dend,self.params['p_som2pyr'])


        result = IC.get_gatingselectivity(self.params)

        ind_sort = np.argsort(result['gs_list'])[::-1]

        Exc1, Exc2 = result['Exc2dend_list']
        Inh1, Inh2 = result['Inh2dend_list']
        Exc1_raw = Exc1/self.params['g_exc']
        Exc2_raw = Exc2/self.params['g_exc']

        Exc1_raw,Exc2_raw,Inh1,Inh2 = \
        [temp.reshape(self.params['n_pyr'],self.params['n_dend_each'])[ind_sort,:].flatten() for temp in [Exc1_raw,Exc2_raw,Inh1,Inh2]]

        self.Exc1_raw, self.Exc2_raw, self.Inh1, self.Inh2 = Exc1_raw, Exc2_raw, Inh1, Inh2


    def plot_Newsome_statespace_legend(self):
        i_dim = 0
        n_m = 6
        for colors in [motion_colors,color_colors]:
            mks = 4
            fig=MyFigure(figsize=(0.75,0.2))
            pl = fig.addplot(rect=[0.05,0.05,0.9,0.9])
            for i_coh in xrange(n_m):
                if i_coh<(n_m/2):
                    mfc = 'white'
                    mec = colors[i_coh]
                else:
                    mfc = colors[n_m-i_coh-1]
                    mec = mfc
                pl.plot(i_coh,0,'o',markerfacecolor=mfc,markeredgecolor=mec,markersize=mks)
            pl.ax.set_frame_on(False)
            pl.ax.set_yticks([])
            pl.ax.set_xticks([])
            fig.save(self.figpath+'plot_Newsome_statespace_legend_%d' % i_dim)
            i_dim += 1


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


    def analytic_twopath(self,p,rate1,rate2):
        '''
        Population of neurons receive input from two pathways, the first path is gated-on
        rate1 and rate2 are the input rate of each pathway
        First we need to convert the input rate into conductance,
        the dend_IO(exc, inh) function takes total excitatory and inhibitory
        conductances as inputs
        '''
        # number of synapses
        num_syn = 15
        g_exc = p['g_exc']*num_syn
        # gating variable
        s1 = MCM.meansNMDA(rate1)
        s2 = MCM.meansNMDA(rate2)

        # Total conductance input
        Exc1 = self.Exc1_raw*s1*g_exc # nS
        Exc2 = self.Exc2_raw*s2*g_exc # nS

        Exc = Exc1+Exc2

        #frac_proj = 0.1 # fraction projection
        N_proj = p['frac_proj']*self.params['n_pyr']
        N_proj0 = np.floor(N_proj)
        N_proj0 = min((N_proj0,self.params['n_pyr']-1))
        N_proj0 = max((N_proj0,0))

        DendV = dend_IO(Exc[:(N_proj0+1)*self.params['n_dend_each']],
                            self.Inh1[:(N_proj0+1)*self.params['n_dend_each']])
        meanDendV = DendV.reshape(N_proj0+1,self.params['n_dend_each']).mean(axis=1)
        SomaR = soma_fv(meanDendV)

        # Make sure firing rate depend smoothly on frac_proj
        rboth = (SomaR[:N_proj0].sum()+SomaR[N_proj0]*(N_proj-N_proj0))/N_proj

        return rboth


    def perform_func(self,x):
        # input x must have unit pA
        y = 1/(1+np.exp(-x/0.99)) # logistic function
        return y

    def quick_Newsome_model(self,p,m,c):
        '''
        Run model for Newsome task with analytic rate units and logistic decision
        m,c are the motion and color coherence, always consider motion context
        '''
        # External input targeting the mixed-sensory layer
        input1 = self.analytic_twopath(p,40*(1+m),40*(1+c)) # return Hz
        input2 = self.analytic_twopath(p,40*(1-m),40*(1-c))

        # Mixed-sensory target the decision layer
        input_diff_current = (input1-input2)*p['sen2dec_exc'] # input the difference of current (pA)
        performance = self.perform_func(input_diff_current)
        return performance

    def quick_runNewsome_psychometric(self,p,savenew=True,savename=''):
        datacolor3 = np.array([150,150,150])/255
        modelcolor3 = np.array([37,37,37])/255

        m_options = np.array([-0.5,-0.15,-0.05,0.05,0.15,0.5])
        c_options = np.array([-0.5,-0.18,-0.06,0.06,0.18,0.5])
        n_options = 6
        mks = 5

        n_plot = 51
        coh_plot = np.linspace(-0.6,0.6,n_plot)
        performance_plot_m = np.zeros((n_options,n_plot))
        for i_c in xrange(n_options):
            for i_m in xrange(n_plot):
                c = c_options[i_c]
                m = coh_plot[i_m]
                performance = self.quick_Newsome_model(p,m,c)
                performance_plot_m[i_c,i_m] += performance*100

        fig=MyFigure(figsize=(4.5,1.7))
        pl = fig.addplot(rect=[0.1,0.3,0.24,0.6])
        pl.plot(m_options*100,self.p_m_m*100,'o',color=datacolor3,markeredgecolor=datacolor3,label='Data',markersize=mks)
        pl.plot(coh_plot*100,performance_plot_m.mean(axis=0),color = modelcolor3,label='Model')
        pylab.ylim([-5,105])
        pylab.yticks([0,25,50,75,100])
        pylab.xticks([-50,-15,15,50],['-50\n Left','-15','15','50\n Right'])
        pylab.ylabel('Choices to right (%)')
        pylab.xlabel('Motion coherence')
        leg = pylab.legend(loc=2,numpoints=1, bbox_to_anchor=(-0.03, 1.15))
        leg.draw_frame(False)

        m_options = c_options[:]
        performance_plot_c = np.zeros((n_options,n_plot))
        for i_m in xrange(n_options):
            for i_c in xrange(n_plot):
                c = coh_plot[i_c]
                m = m_options[i_m]
                performance = self.quick_Newsome_model(p,m,c)
                performance_plot_c[i_m,i_c] += performance*100

        pl = fig.addplot(rect=[0.44,0.3,0.24,0.6])
        pl.plot(c_options*100,self.p_m_c*100,'o',color=datacolor3,markeredgecolor=datacolor3,label='Data',markersize=mks)
        pl.plot(coh_plot*100,performance_plot_c.mean(axis=0),color = modelcolor3,label='Model')
        pylab.ylim([-5,105])
        pylab.yticks([0,25,50,75,100],[])
        pylab.xticks([-50,-15,15,50],['-50\n Red','-15','15','50\n Green'])
        pylab.xlabel('Color coherence')
        pylab.ylabel('Choices to green (%)')
        #title('Motion Context')
        leg = pylab.legend(loc=2,numpoints=1, bbox_to_anchor=(-0.03, 1.15))
        leg.draw_frame(False)

        pl = fig.addplot(rect=[0.73,0.3,0.24,0.6])
        for i_m in xrange(n_options//2):
            pl.plot(coh_plot*100,(performance_plot_c[i_m,:]+performance_plot_c[-1-i_m,:])/2,color=motion_colors4[-i_m-1])
        pylab.ylim([-5,105])
        pylab.yticks([0,25,50,75,100],[])
        pylab.xticks([-50,-15,15,50],['-50\n Red','-15','15','50\n Green'])
        leg = pylab.legend(['Strong','Medium','Weak'],title='Motion coherence',loc=2,bbox_to_anchor=(-0.1, 1.2))
        leg.draw_frame(False)
        pylab.xlabel('Color coherence')

        if savenew:
            fig.save(self.figpath+'CompareWithNewsomePsychometric_'+savename+self.monkey+self.version)

        combined_panel = True
        if combined_panel:
            mks = 3
            fig=MyFigure(figsize=(1.85,1.5))
            pl = fig.addplot(rect=[0.3,0.25,0.6,0.55])
            pl.plot(c_options*100,self.p_m_c*100,'o',color=np.array([8,29,88])/255,
                    markeredgecolor=np.array([8,29,88])/255,markersize=mks,label='Data - Color context')
            pl.plot(coh_plot*100,performance_plot_c.mean(axis=0),color = np.array([8,29,88])/255)
            pl.plot(m_options*100,self.p_m_m*100,'o',color=np.array([65,182,196])/255,
                    markeredgecolor=np.array([65,182,196])/255,label='Data - Motion context',markersize=mks)
            pl.plot(coh_plot*100,performance_plot_m.mean(axis=0),color = np.array([65,182,196])/255,label='Model')
            pylab.ylim([-5,105])
            pylab.yticks([0,25,50,75,100])
            pylab.xticks([-50,-15,15,50],['-50','-15','15','50'])
            pylab.ylabel('Choices to right (%)')
            pylab.xlabel('Motion coherence to right')
            leg = pylab.legend(loc=2,numpoints=1, bbox_to_anchor=(-0.03, 1.35))
            leg.draw_frame(False)
            fig.save(self.figpath+'CompareWithNewsomePsychometric_combpanel_'+savename+self.monkey+self.version)


    def get_sqe_above(self,high_p,rich_return=False):
        p = dict()
        p['g_exc'] = high_p[0] # nS
        p['sen2dec_exc'] = high_p[1] # pA/Hz
        if len(high_p)>2:
            p['frac_proj'] = high_p[2]
        else:
            p['frac_proj'] = 0.1

        win1_all = np.zeros((6,6))
        for i_m in xrange(len(self.m_plot)):
            for i_c in xrange(len(self.c_plot)):
                m = self.m_plot[i_m]
                c = self.c_plot[i_c]
                win1_all[i_m,i_c] = self.quick_Newsome_model(p,m,c)

        sqe = 0
        sqe += sum((self.p_m_m-win1_all.mean(axis=1))**2)
        sqe += sum((self.p_m_c-win1_all.mean(axis=0))**2)

        if rich_return:
            return p
        else:
            return sqe

    def plot_compare(self,p,savenew=True,savename=''):
        cohs = [0.05,0.15,0.5]
        x_plot = [-coh for coh in cohs[::-1]]+cohs
        x_plot = np.array(x_plot)

        p_frac1 = p.copy()
        p_frac1['frac_proj'] = 1
        r_mc_list = np.zeros((6,6))
        for i_m in xrange(len(x_plot)):
            for i_c in xrange(len(x_plot)):
                m = x_plot[i_m]
                c = x_plot[i_c]
                #for a particular value of motion and color m,c \in (-1,1)
                #the input from the two pathways to the four considered populations are
                #(m,c)
                #(-m,c)
                #(m,-c)
                #(-m,-c)
                r_mc_list[i_m,i_c] = self.analytic_twopath(p_frac1,40*(1+m),40*(1+c))

        input_plot_all = r_mc_list - r_mc_list[::-1,::-1]
        win1_list = self.perform_func(input_plot_all)

        n_c = n_m = 6

        Dim_motion = r_mc_list + r_mc_list[:,::-1] - r_mc_list[::-1,:] - r_mc_list[::-1,::-1]
        Dim_color = r_mc_list - r_mc_list[:,::-1] + r_mc_list[::-1,:] - r_mc_list[::-1,::-1]

        smalllinewidth = 1
        largelinewidth = 2.5

        Dim_motion = Dim_motion.mean(axis=1)

        fig=MyFigure(figsize=(4,4))
        pl = fig.addplot(rect=[0.2,0.2,0.7,0.7])
        for i_motion in xrange(n_m):
            if i_motion<(n_m/2):
                facecolor = 'white'
                linewidth = smalllinewidth
                linecolor = motion_colors[i_motion]
                px = 1
            else:
                linecolor = motion_colors[n_m-i_motion-1]
                facecolor = motion_colors[n_m-i_motion-1]
                linewidth = largelinewidth
                px = -1
            pl.plot(px,Dim_motion[i_motion],
            'o',markeredgecolor=linecolor,
            markerfacecolor=facecolor,linewidth=linewidth)


        Dim_color1 = Dim_color[:3,:].mean(axis=0)
        Dim_color2 = Dim_color[3:,:].mean(axis=0)

        print Dim_motion
        print Dim_color1

        for i_color in xrange(n_c):
            if i_color<(n_c/2):
                linecolor = color_colors[i_color]
                markerfacecolor = 'white'
            else:
                linecolor = color_colors[n_c-i_color-1]
                markerfacecolor = linecolor
            pl.plot(0.5,Dim_color1[i_color],'o',
                    markeredgecolor=linecolor,markerfacecolor=markerfacecolor,linewidth=smalllinewidth)
            pl.plot(-0.5,Dim_color2[i_color],'o',
                    markeredgecolor=linecolor,markerfacecolor=markerfacecolor,linewidth=largelinewidth)
        pylab.xlabel('Dim Dec')
        pylab.ylabel('Dim Motion/Color')

        self.quick_runNewsome_psychometric(p,savenew=savenew,savename=savename)


    def run_Newsome_model(self,p,m,c, biphasic_input=False):
        #m = -0.05
        #c = -0.5
        '''
        Run model for Newsome task with all rate units
        m,c are the motion and color coherence, always consider motion context
        '''
        # External input targeting the mixed-sensory layer
        p_proj = p.copy()
        p['frac_proj'] = 1
        input_mc = self.analytic_twopath(p,40*(1+m),40*(1+c))
        input_none = self.analytic_twopath(p,40*(1-m),40*(1-c))
        input_m = self.analytic_twopath(p,40*(1+m),40*(1-c))
        input_c = self.analytic_twopath(p,40*(1-m),40*(1+c))

        input_mc_proj = self.analytic_twopath(p_proj,40*(1+m),40*(1+c))
        input_none_proj = self.analytic_twopath(p_proj,40*(1-m),40*(1-c))

        # Mixed-sensory target the decision layer
        p_WW = dict()
        p_WW['Ttotal'] = 0.8
        p_WW['dt'] = 0.2/1000
        p_WW['Tstim'] = -1
        p_WW['n_trial'] = 100
        p_WW['record_dt'] = 0.05
        p_WW['mu0'] = 30


        if biphasic_input:
            '''
            tau_rise = 0.2
            tau_decay = 0.3
            maxamp = 1.5
            NT = int(p_WW['Ttotal']/p_WW['dt'])
            tplot = np.arange(0,NT)*p_WW['dt']
            amp = (np.exp(-tplot/tau_decay)-np.exp(-tplot/tau_rise))
            ratio = 1/max(amp)*maxamp
            amp = amp*ratio

            N_record = int(p_WW['Ttotal']/p_WW['record_dt'])
            tplot = np.arange(0,N_record)*p_WW['record_dt']
            amp_record = (np.exp(-tplot/tau_decay)-np.exp(-tplot/tau_rise))*ratio
            '''

            NT = int(p_WW['Ttotal']/p_WW['dt'])
            tplot = np.arange(0,NT)*p_WW['dt']
            tau_syn = 0.1
            tau_decay = 0.2
            maxamp = 1.0
            kernel = np.exp(-tplot/tau_syn)
            amp = scipy.signal.fftconvolve(np.exp(-tplot/tau_decay),kernel)
            amp = amp[:len(tplot)]
            ratio = 1/max(amp)*maxamp
            amp = amp*ratio + 0.5 # add a constant input

            N_record = int(p_WW['Ttotal']/p_WW['record_dt'])
            amp_record = amp[range(0,NT,NT//N_record)]
        else:

            NT = int(p_WW['Ttotal']/p_WW['dt'])
            tplot = np.arange(0,NT)*p_WW['dt']
            amp = (tplot>0.01)*(tplot<p_WW['Ttotal']-0.06)

            N_record = int(p_WW['Ttotal']/p_WW['record_dt'])
            tplot = np.arange(0,N_record)*p_WW['record_dt']
            #amp_record = (tplot>0.01)*(tplot<p_WW['Ttotal']-0.06)
            amp_record = tplot>0.01


        diff_current = (input_mc_proj-input_none_proj)*p['sen2dec_exc']*amp

        p_WW['diff_current'] = diff_current


        model = WW.WongWangModel(p_WW)
        model.run(record=True)
        valid_trials = any([model.r1>10,model.r2>10],axis=0)
        if m>0:
            correct_trials = model.r1>model.r2
        else:
            correct_trials = model.r1<model.r2

        r1_record = model.r1_record[valid_trials*correct_trials,:].mean(axis=0)
        r2_record = model.r2_record[valid_trials*correct_trials,:].mean(axis=0)

        result = dict()
        result['r1'] = r1_record
        result['r2'] = r2_record
        result['input_mc'] = input_mc*amp_record
        result['input_none'] = input_none*amp_record
        result['input_m'] = input_m*amp_record
        result['input_c'] = input_c*amp_record

        return result


    def plot_state_space(self,p,savenew=True):
        '''
        Plot the state-space plot for fitted model
        '''
        n_list = 6
        record_list = [np.array([])]*n_list
        name_list = ['r1','r2','input_mc','input_none','input_m','input_c']

        for i_m in xrange(len(self.m_plot)):
            for i_c in xrange(len(self.c_plot)):
                m = self.m_plot[i_m]
                c = self.c_plot[i_c]
                result = self.run_Newsome_model(p,m,c, biphasic_input=False)
                for i_list in xrange(n_list):
                    record_list[i_list] = np.concatenate((record_list[i_list],result[name_list[i_list]]))
                print 'm = %0.2f, c = %0.2f' % (m,c)

        # z-score results
        zscore_list = list()
        for i_list in xrange(n_list):
            zscore = record_list[i_list]
            temp = zscore[zscore>0.01] # exclude the data before and after dots
            #zscore -= np.mean(record_list[i_list])
            zscore = zscore/np.std(temp)
            zscore_list.append(zscore)
        zs = zscore_list

        Dim_dec = -(zs[0]-zs[1])/2
        Dim_motion = (zs[2]-zs[3]+zs[4]-zs[5])/4
        Dim_color = (zs[2]-zs[3]-zs[4]+zs[5])/4

        N_t = len(result['r1'])

        maxr = max([max(abs(Dim_dec)),max(abs(Dim_motion)),max(abs(Dim_color))])
        smalllinewidth = 0.5
        largelinewidth = 1
        mks = 2

        n_m = 6
        n_c = 6
        px = [0]*n_m
        py = [0]*n_m
        for i_cond in xrange(n_m*n_c):
            i_motion = i_cond//n_m
            px[i_motion] += Dim_dec[i_cond*N_t:(i_cond+1)*N_t]/n_c
            py[i_motion] += Dim_motion[i_cond*N_t:(i_cond+1)*N_t]/n_c

        dotoffcolor = np.array([84,39,143])/255.
        fig=MyFigure(figsize=(3,1.5))
        pl = fig.addplot(rect=[0.01,0.05,0.45,0.9])
        for i_motion in xrange(n_m):
            if i_motion<(n_m/2):
                mfc = 'white'
                linewidth = smalllinewidth
                linecolor = motion_colors[i_motion]
                mec = linecolor
            else:
                linecolor = motion_colors[n_m-i_motion-1]
                mfc = motion_colors[n_m-i_motion-1]
                linewidth = largelinewidth
                mec = mfc
            pl.plot([px[i_motion][-1]]*2,[py[i_motion][-1],0],
            'o-',color=dotoffcolor,
            markerfacecolor=dotoffcolor,markeredgecolor=dotoffcolor,linewidth=linewidth,markersize=mks)

            pl.plot(px[i_motion],py[i_motion],
            'o-',color=linecolor,
            markerfacecolor=mfc,markeredgecolor=mec,linewidth=linewidth,markersize=mks)

        #xlabel('Dim Dec')
        #ylabel('Dim Motion')
        pylab.xlim((-maxr,maxr))
        pylab.ylim((-maxr,maxr))
        pylab.axis('equal')
        pl.ax.set_frame_on(False)
        pl.ax.set_yticks([])
        pl.ax.set_xticks([])

        px_1 = [0]*n_c
        py_1 = [0]*n_c
        px_2 = [0]*n_c
        py_2 = [0]*n_c
        for i_cond in xrange(n_m*n_c):
            i_color = np.mod(i_cond,n_c)
            i_motion = i_cond//n_m
            if i_motion<(n_m/2):
                px_1[i_color] += Dim_dec[i_cond*N_t:(i_cond+1)*N_t]/n_c*2
                py_1[i_color] += Dim_color[i_cond*N_t:(i_cond+1)*N_t]/n_c*2
            else:
                px_2[i_color] += Dim_dec[i_cond*N_t:(i_cond+1)*N_t]/n_c*2
                py_2[i_color] += Dim_color[i_cond*N_t:(i_cond+1)*N_t]/n_c*2

        pl = fig.addplot(rect=[0.54,0.05,0.45,0.9])
        for i_color in xrange(n_c):
            if i_color<(n_c/2):
                linecolor = color_colors[i_color]
                mfc = 'white'
                mec = linecolor
            else:
                linecolor = color_colors[n_c-i_color-1]
                mfc = linecolor
                mec = mfc
            pl.plot([px_1[i_color][-1]]*2,[py_1[i_color][-1],0],'o-',
                    color=dotoffcolor,markerfacecolor=dotoffcolor,markeredgecolor=dotoffcolor,linewidth=smalllinewidth,markersize=mks)
            pl.plot([px_2[i_color][-1]]*2,[py_2[i_color][-1],0],'o-',
                    color=dotoffcolor,markerfacecolor=dotoffcolor,markeredgecolor=dotoffcolor,linewidth=largelinewidth,markersize=mks)


            pl.plot(px_1[i_color],py_1[i_color],'o-',
                    color=linecolor,markerfacecolor=mfc,markeredgecolor=mec,linewidth=smalllinewidth,markersize=mks)
            pl.plot(px_2[i_color],py_2[i_color],'o-',
                    color=linecolor,markerfacecolor=mfc,markeredgecolor=mec,linewidth=largelinewidth,markersize=mks)
        #xlabel('Dim Dec')
        #ylabel('Dim Color')
        pylab.xlim((-maxr,maxr))
        pylab.ylim((-maxr,maxr))
        pl.ax.set_frame_on(False)
        pl.ax.set_yticks([])
        pl.ax.set_xticks([])
        if savenew:
            fig.save(self.figpath+'plot_Newsome_statespace'+'_'+self.monkey+self.version)

    def update_monkey_info(self, monkey='F'):
        # Monkey A performance
        self.monkey = monkey
        if self.monkey is 'A':
            self.p_m_m = np.array([0,0.13,0.36,0.62,0.87,0.97]) # motion context, motion coherence
            self.p_m_c = np.array([0.41,0.47,0.48,0.50,0.51,0.55]) # motion context, color coherence
            #self.p_m_c = 0.5*np.ones(6) # motion context, color coherence
            self.p0 = [1.8,4.4,0.08]
            self.p_bounds = ((0.1,5),(0.01,30),(0.05,0.2))

            #p0, p_bounds = p0[:2], p_bounds[:2]
        elif self.monkey is 'F':
            self.p_m_m = np.array([0.095,0.226,0.355,0.566,0.756,0.940])
            self.p_m_c = np.array([0.346,0.416,0.439,0.477,0.482,0.644])
            self.p0 = [1.6,15,0.1]
            self.p_bounds = ((0.1,5),(0.01,30),(0.05,0.5))
        else:
            ValueError('Unknown monkey name')
        Correctmean = True
        if Correctmean:
            self.p_m_m = self.p_m_m - np.mean(self.p_m_m) + 0.5
            self.p_m_c = self.p_m_c - np.mean(self.p_m_c) + 0.5

        self.m_plot = np.array([-0.5,-0.15,-0.05,0.05,0.15,0.5])
        self.c_plot = np.array([-0.5,-0.18,-0.06,0.06,0.18,0.5])


    def fit_run_model(self, monkey='F', savenew=True, plotting=True, savename=''):
        self.update_monkey_info(monkey=monkey)

        r = scipy.optimize.minimize(lambda p: self.get_sqe_above(p),self.p0, method='SLSQP',bounds=self.p_bounds)
        #print r
        p = self.get_sqe_above(r.x, rich_return=True)

        if plotting:
            self.plot_compare(p,savenew=savenew,savename=savename)
            #self.plot_state_space(p,savenew=savenew)
        return r

    def direct_fitdata(self, monkey='F'):
        self.update_monkey_info(monkey=monkey)

        def get_sqe_direct(x_wid, x_plot, y_target):
            y_plot = 1/(1+np.exp(-x_plot/x_wid)) # logistic function
            sqe = np.sum((y_target-y_plot)**2)
            return sqe

        r1 = scipy.optimize.minimize(lambda x_wid: get_sqe_direct(x_wid,self.m_plot,self.p_m_m),[0.5], method='SLSQP',bounds=[(0.01,2)])
        r2 = scipy.optimize.minimize(lambda x_wid: get_sqe_direct(x_wid,self.c_plot,self.p_m_c),[0.5], method='SLSQP',bounds=[(0.01,2)])
        best_sqe = r1.fun + r2.fun

        return best_sqe

    def fit_vary_param(self,monkey='F'):
        # Vary network parameters while fitting
        with open(self.datapath+'fit_vary_param'+ self.version,'rb') as f:
            res = pickle.load(f)
        
        n_som2dend_list = range(2,21)
        seed_list = range(70)
        for n_som2dend in n_som2dend_list:
            if n_som2dend in res.keys():
                info = res[n_som2dend]
            else:
                info = dict()
            for seed in seed_list:
                if seed not in info.keys():
                    self.set_EI(n_som2dend=n_som2dend,seed=seed)
                    r = self.fit_run_model(monkey=monkey,savenew=False,plotting=False)
                    print 'n_som2dend = %d, seed=%d, sqe=%0.4f' % (n_som2dend,seed,r.fun)                
                    info[seed] = r.fun
            res[n_som2dend] = info
            
        res['best_sqe'] = self.direct_fitdata(monkey=monkey)
        self.set_EI(n_som2dend=n_som2dend,seed=seed,no_selectivity=True)
        r = self.fit_run_model(monkey=monkey,savenew=False,plotting=False)
        res['worst_sqe'] = r.fun
        res['monkey'] = monkey
        with open(self.datapath+'fit_vary_param'+ self.version,'wb') as f:
            pickle.dump(res,f)

        return res

    def plot_fit_vary_param(self):
        with open(self.datapath+'fit_vary_param'+ self.version,'rb') as f:
            res = pickle.load(f)

        n_som2dend_list = res.keys()
        n_som2dend_list.remove('best_sqe')
        n_som2dend_list.remove('worst_sqe')
        n_som2dend_list.remove('monkey')

        sqe_list = list()
        for n_som2dend in n_som2dend_list:
            sqes = res[n_som2dend].values()
            sqe_list.append(np.median(sqes))
            
        fig=MyFigure(figsize=(2,1.7))
        pl = fig.addplot(rect=[0.3,0.3,0.65,0.65])
        pl.plot(n_som2dend_list,sqe_list,
            'o-',markeredgecolor='black',markerfacecolor='black',markersize=2,
            color=np.array([150,150,150])/255,linewidth=1.5,label='model')

        pl.plot(n_som2dend_list,[res['best_sqe']]*len(n_som2dend_list),
            '-',markeredgecolor='black',markerfacecolor='black',
            color=np.array([150,150,150])/255,linewidth=1.5,label='optimal')

        pylab.ylabel('Sum of squared errors')
        pylab.xlabel(r'$N_{\mathit{SOM}\rightarrow dend}$')
        #pl.ax.set_yticks([0.01,0.02,0.03])
        pl.ax.set_yticks([0,0.05,0.1])
        pl.ax.set_xticks([5,10,15,20])
        pylab.legend(loc=2,numpoints=1,frameon=False)

        #pl.plot(n_som2dend_list,[res['worst_sqe']]*len(n_som2dend_list),'--',markeredgecolor='black',markerfacecolor='black',linewidth=1.5)

        fig.save(self.figpath+ 'NewsomeFit_vary_param' + self.version)

