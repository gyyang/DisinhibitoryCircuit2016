# -*- coding: utf-8 -*-
"""
Created on Thu May  1 13:43:49 2014

@author: guangyuyang

Wong & Wang 2006
"""
from __future__ import division
import datetime
import os
import time
from pylab import *
import numpy.random



def F(I, a=270., b=108., d=0.154):
    """F(I) for vector I"""
    return (a*I - b)/(1.-np.exp(-d*(a*I - b)))

class WongWangModel():
    def __init__(self, outsidePara=dict()):
        p = dict()
        p['gE'] = 0.2609
        p['gI'] = -0.0497 # cross-inhibition strength [nA]
        p['I0'] = 0.3255 # background current [nA]
        p['tauS'] = 0.1 # Synaptic time constant [sec]
        p['gamma'] = 0.641 # Saturation factor for gating variable
        p['tau0'] = 0.002 # Noise time constant [sec]
        p['sigma'] = 0.02 # Noise magnitude [nA]
        #p['mu0'] = 20. # Stimulus firing rate [Hz]
        p['Jext'] = 0.52 # Stimulus input strength [pA/Hz]
        p['Ttotal'] = 2. # Total duration of simulation [sec]
        p['Tstim'] = 0.1 # Time of stimulus onset [sec]
        p['dt'] = 0.0005 # Simulation time step [sec]
        p['n_trial'] = 1 # number of trials run
        p['record_dt'] = 0.05
        self.params = p
        for key in outsidePara:
            self.params[key] = outsidePara[key] # overwrite the old value
        p['mean_current'] = p['mu0']*p['Jext'] # [pA]

    def run(self, record=True):
        p = self.params

        # Set random seed
        #np.random.seed(10)

        # Number of time points
        NT = int(p['Ttotal']/p['dt'])

        mean_stim = ones(NT)*p['mean_current']/1000 # [nA]
        #diff_stim = (p['Tstim']<tplot)*p['Jext']*p['mu0']*p['coh']/100.*2
        self.Istim1_plot = mean_stim + p['diff_current']/2/1000 # [nA]
        self.Istim2_plot = mean_stim - p['diff_current']/2/1000

        # Initialize S1 and S2
        S1 = 0.1*ones(p['n_trial'])
        S2 = 0.1*ones(p['n_trial'])

        Ieta1 = zeros(p['n_trial'])
        Ieta2 = zeros(p['n_trial'])

        if record:
            n_record = int(p['record_dt']//p['dt'])
            i_record = 0
            N_record = int(p['Ttotal']/p['record_dt'])
            self.r1_record = zeros((p['n_trial'],N_record))
            self.r2_record = zeros((p['n_trial'],N_record))

        # Loop over time points in a trial
        for i_t in xrange(NT):
            # Random dot stimulus
            Istim1 = self.Istim1_plot[i_t]
            Istim2 = self.Istim2_plot[i_t]


            # Total synaptic input

            Isyn1 = p['gE']*S1 + p['gI']*S2 + Istim1 + Ieta1
            Isyn2 = p['gE']*S2 + p['gI']*S1 + Istim2 + Ieta2

            # Transfer function to get firing rate

            r1  = F(Isyn1)
            r2  = F(Isyn2)

            #---- Dynamical equations -------------------------------------------

            # Mean NMDA-mediated synaptic dynamics updating
            S1_next = S1 + p['dt']*(-S1/p['tauS'] + (1-S1)*p['gamma']*r1)
            S2_next = S2 + p['dt']*(-S2/p['tauS'] + (1-S2)*p['gamma']*r2)

            # Ornstein-Uhlenbeck generation of noise in pop1 and 2
            Ieta1_next = Ieta1 + (p['dt']/p['tau0'])*(p['I0']-Ieta1) + sqrt(p['dt']/p['tau0'])*p['sigma']*numpy.random.randn(p['n_trial'])
            Ieta2_next = Ieta2 + (p['dt']/p['tau0'])*(p['I0']-Ieta2) + sqrt(p['dt']/p['tau0'])*p['sigma']*numpy.random.randn(p['n_trial'])

            S1 = S1_next
            S2 = S2_next
            Ieta1 = Ieta1_next
            Ieta2 = Ieta2_next

            if record:
                if mod(i_t,n_record) == 1:
                    self.r1_record[:,i_record] = r1
                    self.r2_record[:,i_record] = r2
                    i_record += 1


        self.r1 = r1
        self.r2 = r2


        
def test_model():
    p = dict()
    p['coh'] = 100
    p['Tstim'] = 0.1
    p['Ttotal'] = 1.0
    p['dt'] =0.5/1000
    figure()
    start = time.clock()
    for i_trial in xrange(10):
        model = WongWangModel(p)
        model.run()
        print 'time spend %0.3f s' % (time.clock()-start)
        #os.system('say "Go bender, go bender!"')
        plot(model.tplot,model.r1smooth,'black')
        plot(model.tplot,model.r2smooth,'red')

#test_model()