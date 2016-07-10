'''2014-12-08
Final round. The spiking model as a class.
'''

from __future__ import division
import time
import os
import re
import numpy.random
import bisect
import scipy as sp
import random as pyrand
import brian_no_units
from brian import *

def read_params(filename, skipchars=['#']):
    with open(filename, 'rU') as f:
        params = dict()
        for line in f:
            line = line.strip()
            if not line or line[0] in skipchars:
                continue

            p = line.split()
            params[p[0]] = eval('*'.join(p[1:]))

    return params

def read_eqs(filename):
    eqs = dict()
    name = ''
    with open(filename, 'rU') as f:
        for line in f:
            line = line.strip()

            if name:
                if line:
                    eqs[name] += line + '\n'
                else:
                    name = ''
            elif line:
                match = re.match(r'\[\s*(.*?)\s*\]', line)
                if match is not None:
                    name = match.groups(1)[0]
                    eqs[name] = ''

    return eqs

def accelerate():
    set_global_preferences(useweave=True,
                           usecodegen=True,
                           usecodegenweave=True,
                           usecodegenstateupdate=True,
                           usenewpropagate=True,
                           usecodegenthreshold=True,
                           gcc_options=['-ffast-math', '-march=native']
                           )

def SynapseChange(wpre,AlphaP,AlphaD,p):
    '''
    Model from Graupner & Brunel 2012
    Original formulation
    wpre, strength before learning
    alphaP, fraction of time above potentiation threshold
    alphaD, fraction of time above potentiation threshold
    T, amount of time spent
    
    Turned out T, does not matter. All that matter are 
    AlphaP = alphaP*T, AlphaD = alphaD*T
    AlphaP and AlphaD have units: second
    They represent the total amount of time above respective threshold
    
    p: parameter dictionary
    '''   
        
    beta = (p['w1']-wpre)/(p['w1']-p['w0'])

    if AlphaD<1e-7 and AlphaP<1e-7:
        wpost = wpre
    else:
        GammaD = p['gammaD']*AlphaD
        GammaP = p['gammaP']*AlphaP
        rhoBar = GammaP/(GammaP+GammaD)
        sigmaRhoSquare = p['sigma']**2 * (AlphaD+AlphaP)/(GammaP+GammaD)

        exptemp = exp(-(GammaD+GammaP)/p['tau'])
        rho0 = 0
        temp = -(p['rhoStar']-rhoBar+(rhoBar-rho0)*exptemp)
        temp = temp/sqrt(sigmaRhoSquare*(1-exptemp**2))
        down2upProb = 0.5*(1+math.erf(temp))
        down2downProb = 1-down2upProb

        rho0 = 1
        temp = -(p['rhoStar']-rhoBar+(rhoBar-rho0)*exptemp)
        temp = temp/sqrt(sigmaRhoSquare*(1-exptemp**2))
        up2upProb = 0.5*(1+math.erf(temp))
        up2downProb = 1-up2upProb

        wpost = (beta*down2downProb+(1-beta)*up2downProb)*p['w0'] + \
        (beta*down2upProb + (1-beta)*up2upProb)*p['w1']

    return wpost

def SynapseChangeWithHomeostatis(wpre,AlphaP,AlphaD,ZetaP,ZetaD,p):
    '''
    Model from Graupner & Brunel 2012
    Original formulation
    wpre, strength before learning
    alphaP, fraction of time above potentiation threshold
    alphaD, fraction of time above potentiation threshold
    T, amount of time spent

    Turned out T, does not matter. All that matter are
    AlphaP = alphaP*T, AlphaD = alphaD*T
    AlphaP and AlphaD have units: second
    They represent the total amount of time above respective threshold

    ZetaP and ZetaD are the time spent in the potentiation and depression zone
    for the homeostatis mechanism
    GammaD and GammaP are modified as
    GammaD = gammaD*AlphaD + homeoD*ZetaD
    p: parameter dictionary
    '''

    beta = (p['w1']-wpre)/(p['w1']-p['w0'])
    GammaD = p['gammaD']*AlphaD + p['homeo_d']*ZetaD
    GammaP = p['gammaP']*AlphaP + p['homeo_p']*ZetaP

    if (GammaD+GammaP)<1e-7:
        wpost = wpre
    else:

        rhoBar = GammaP/(GammaP+GammaD)
        sigmaRhoSquare = p['sigma']**2 * (AlphaD+AlphaP)/(GammaP+GammaD)

        exptemp = exp(-(GammaD+GammaP)/p['tau'])
        rho0 = 0
        temp = -(p['rhoStar']-rhoBar+(rhoBar-rho0)*exptemp)
        temp = temp/sqrt(sigmaRhoSquare*(1-exptemp**2))
        down2upProb = 0.5*(1+math.erf(temp))
        down2downProb = 1-down2upProb

        rho0 = 1
        temp = -(p['rhoStar']-rhoBar+(rhoBar-rho0)*exptemp)
        temp = temp/sqrt(sigmaRhoSquare*(1-exptemp**2))
        up2upProb = 0.5*(1+math.erf(temp))
        up2downProb = 1-up2upProb

        wpost = (beta*down2downProb+(1-beta)*up2downProb)*p['w0'] + \
        (beta*down2upProb + (1-beta)*up2upProb)*p['w1']

    return wpost

def meansNMDA(rate,p=None):
    if p is None:
        temp = 0.06*rate
        # default parameter value
        # 0.06 is obtained from tau_rise = 2ms, tau_decay=100ms, alpha=0.3 /ms
    else:
        x_mean = rate*p['tauNMDARise']
        temp = p['alphaNMDA']*p['tauNMDADecay']*x_mean
    s_mean = temp/(1+temp)
    return s_mean

def dend_IO(exc, inh):
    '''
    for a single dendritic branch, its input-output relationship given current
    excitation and inhibition

    dendrite model is fitted from simulation data, see CurveFitting.py


    :param p: parameter dictionary
    :param exc: the vector of total excitation (nS)
    :param inh: the vector of total inhibition (nS)
    :return: voltage vector given excitation and inhibition
    actually depolarization with respect to reveral potential -70mV
    '''

    # Fitted from data in data/WeightVsInhibition2014-12-01_5
    #q = [ 5.73816701,  8.476012,    6.59250773,  1.57537614]

    # Fitted from data in data/paramV1/DendVvsgErI
    q = [5.5592051,   9.64361681,  6.5386557,   0.7779744]

    mid_point = q[0]*(4+inh) # 4nS, leak conductance of dendrites

    spread = q[1]*exp(inh/q[2])
    
    v = 30*(1+tanh((exc-mid_point)/spread)) + q[3] -70

    return v

def soma_fv(v):
    # v is averaged dendritic voltage
    dv = v + 70
    # Fitted from data in data/paramV1/RatevsDendV
    fr = 2.59035567 + 0.13104463*(dv**2.07809856)
    return fr

def soma_fI(I):
    # I is the injection current (pA)
    # Fitted from data in data/paramV1/RatevsI
    fr = ((I+174.85951028)*(I>-174.85951028)/45.16030095)**2.89003711
    return fr

def soma_IO(p, dend_v):
    '''
    for a multi-branch neuron, given the dendritic voltage, return the somatic
    activity
    :param p: parameter dictionary
    :param dend_v: vector of dendritic voltage, one voltage for each dendrite
    :return:
    '''
    # firing rate

    # mean dendritic voltage for each neuron
    vdendmean = dend_v.reshape((p['num_soma'],p['num_DendEach'])).mean(axis=1)
    # notice the following equation is calculated for 10 dendrites
    # if there are N dendrites the voltage should be multiplied by N/10

    #vdendmean_equiv = vdendmean/10*p['num_DendEach']
    vdendmean_equiv = vdendmean
    # firing rate
    # Low firing parameter set
    #fr = 0.003911*(vdendmean+1)**2.719
    # High firing parameter set
    fr = soma_fv(vdendmean_equiv)
    return fr

def crossthresholdtime(x,theta):
    '''
    :param x: trace of value
    :param theta: threshold
    :return: smoothed time of x crossing threshold theta

    Original implementation to get the amount of time above threshold
    However, this results in non-smooth objective function, making the optimization
    much harder. In the new method, I smoothened the objective function.
    AlphaP = sum(CaTrace>thetaP)*p['dt']*p['repeat_times']
    AlphaD = sum(CaTrace>thetaD)*p['dt']*p['repeat_times']
    '''
    ind = bisect.bisect_left(x,theta)
    N = len(x)
    if ind == 0:
        if theta > (x[0]-(x[1]-x[0])/2.0):
            alpha = (theta-(x[0]-(x[1]-x[0])/2.0))/(x[1]-x[0])
        else:
            alpha = 0
    elif ind == len(x):
        if (x[-1]+(x[-1]-x[-2])/2.0)>theta:
            alpha = N - ((x[-1]+(x[-1]-x[-2])/2.0)-theta)/(x[-1]-x[-2])
        else:
            alpha = N
    else:
        alpha = (theta-x[ind-1])/(x[ind]-x[ind-1])+ind-0.5
    alpha = N-alpha
    return alpha

class Model(NetworkOperation):
    def __init__(self, paramsfile, eqsfile, outsidePara=dict(),rndSeed=None):

        # Clock
        self.params = read_params(paramsfile)
        for key in outsidePara:
            self.params[key] = outsidePara[key] # overwrite the old value

        self.simu_clock = Clock(self.params['dt'])
        self.simu_clock_synapse = Clock(self.params['dt'])
        self.record_clock = Clock(dt=self.params['record_dt'])

        NetworkOperation.__init__(self,clock=self.simu_clock)

        self.eqs = read_eqs(eqsfile) # Equations
        # Here both para and eqs are dictionaries themselves

        # Brian network components
        self.network = dict()
        self.connection = dict()
        self.monitor = dict()

        if rndSeed is not None:
            pyrand.seed(324823+rndSeed)
            numpy.random.seed(324823+rndSeed)


    def __call__(self, *args, **kwargs):
        # overwrite original call
        pass


    def make_model(self, num_soma=1, condition = 'invitro',
                   clamped_dendvolt=None,record_all=True):
        n = self.network
        p = self.params
        mon = self.monitor

        p['num_Soma'] = num_soma
        p['num_Dend'] = p['num_DendEach'] * p['num_Soma']

        if condition is 'invivo':
            p['gEachCouple_pyr'] = p['gEachCouple_pyr_vivo']
        p['gTotCouple_pyr'] = p['gEachCouple_pyr'] * p['num_DendEach']

        temp = p['tau_Ca_bAP_rise']/p['tau_Ca_bAP_decay']
        p['norm_factor_Ca'] = temp**(1/(temp-1))

        temp = p['tauGABABRise']/p['tauGABABDecay']
        p['norm_factor_GABAB'] = temp**(1/(temp-1))

        # Defining Neuron Group
        def myreset(P, spikes):
            P.V[spikes] = p['vReset_pyr']
            P.bAPCa_rise[spikes] = P.bAPCa_rise[spikes] + p['norm_factor_Ca']
        def myrefrac(P, spikes):
            P.V[spikes] = p['vReset_pyr']

        # Defining Neuron Group
        n['Soma'] = NeuronGroup(p['num_Soma'], Equations(self.eqs['Soma'], **p),
                                clock=self.simu_clock, threshold=p['vThres_pyr'],
                                reset=CustomRefractoriness(myreset,refracfunc=myrefrac,period=p['refracTime_pyr']))
        n['Soma'].V = p['vRest_pyr']
        n['Soma'].vShadow = p['vRest_pyr']
        n['Soma'].vAveDend = p['vRest_pyr']


        n['Dend'] = NeuronGroup(p['num_Dend'], Equations(self.eqs['Dend'], **p), clock=self.simu_clock)
        n['Dend'].V = p['vRest_pyr']
        n['Dend'].vSoma = p['vRest_pyr']


        @network_operation(when='start',clock=self.simu_clock)
        def SomaDendCoupling():
            temp = n['Dend'].V.reshape((p['num_Soma'],p['num_DendEach']))
            n['Soma'].vAveDend = mean(temp,axis=1)
            n['Dend'].vSoma = n['Soma'].vShadow.repeat(p['num_DendEach'])

        @network_operation(when='start',clock=self.simu_clock)
        def ClampedDend():
            n['Soma'].vAveDend = clamped_dendvolt
            n['Dend'].vSoma = n['Soma'].vShadow.repeat(p['num_DendEach'])

        if clamped_dendvolt is not None:
            self.contained_objects += [ClampedDend]
        else:
            self.contained_objects += [SomaDendCoupling]

        # back-propagating action potential
        n['bAP'] = Connection(n['Soma'], n['Dend'], 'V',weight = lambda i,j: (i==j//p['num_DendEach'])*p['bAP_amp'], delay=p['bAP_delay'])

        if record_all:
            mon['MvDend'] = StateMonitor(n['Dend'],'V',record=True, clock=self.record_clock)
            mon['MvSoma'] = StateMonitor(n['Soma'],'V',record=True, clock=self.record_clock)
            mon['MiNMDA'] = StateMonitor(n['Dend'],'iTotNMDA',record=True, clock=self.record_clock)
        mon['MSpike'] = SpikeMonitor(n['Soma'])

    def make_model_dendrite_only(self, num_soma=1, condition = 'invitro',
                   clamped_somavolt=None):
        n = self.network
        p = self.params
        mon = self.monitor

        p['num_Soma'] = num_soma
        p['num_Dend'] = p['num_DendEach'] * p['num_Soma']

        if condition is 'invivo':
            p['gEachCouple_pyr'] = p['gEachCouple_pyr_vivo']
        p['gTotCouple_pyr'] = p['gEachCouple_pyr'] * p['num_DendEach']

        temp = p['tau_Ca_bAP_rise']/p['tau_Ca_bAP_decay']
        p['norm_factor_Ca'] = temp**(1/(temp-1))


        n['Dend'] = NeuronGroup(p['num_Dend'], Equations(self.eqs['Dend_only'], **p), clock=self.simu_clock)
        n['Dend'].V = p['vRest_pyr']

        if clamped_somavolt is not None:
            n['Dend'].vSoma = clamped_somavolt
        else:
            n['Dend'].vSoma = p['vRest_pyr']

        mon['MvDend'] = StateMonitor(n['Dend'],'V',record=True, clock=self.record_clock)


    def make_model_soma_voltageclamped(self, num_soma=1,
                   clamped_somavolt=+10*mV, store_all=True):
        '''
        In-vitro soma voltage clamp
        Disable spiking
        '''
        n = self.network
        p = self.params
        mon = self.monitor

        p['num_Soma'] = num_soma
        #p['num_DendEach'] = 10
        p['num_Dend'] = p['num_DendEach'] * p['num_Soma']

        p['gTotCouple_pyr'] = p['gEachCouple_pyr'] * p['num_DendEach']

        temp = p['tau_Ca_bAP_rise']/p['tau_Ca_bAP_decay']
        p['norm_factor_Ca'] = temp**(1/(temp-1))

        # Defining Neuron Group
        n['Soma'] = NeuronGroup(p['num_Soma'], Equations(self.eqs['Soma'], **p),
                                clock=self.simu_clock, threshold=+100*mV)
        n['Soma'].V = clamped_somavolt
        n['Soma'].vShadow = clamped_somavolt
        n['Soma'].vAveDend = p['vRest_pyr']


        n['Dend'] = NeuronGroup(p['num_Dend'], Equations(self.eqs['Dend'], **p), clock=self.simu_clock)
        n['Dend'].V = p['vRest_pyr']
        n['Dend'].vSoma = p['vRest_pyr']

        @network_operation(when='start',clock=self.simu_clock)
        def SomaDendCoupling():
            temp = n['Dend'].V.reshape((p['num_Soma'],p['num_DendEach']))
            n['Soma'].vAveDend = mean(temp,axis=1)
            n['Dend'].vSoma = n['Soma'].vShadow.repeat(p['num_DendEach'])

        @network_operation(when='start',clock=self.simu_clock)
        def ClampedSoma():            
            n['Soma'].vShadow = clamped_somavolt
            n['Soma'].V = clamped_somavolt

        self.contained_objects += [SomaDendCoupling, ClampedSoma]

        mon['MvDend'] = StateMonitor(n['Dend'],'V',record=True, clock=self.record_clock)
        mon['MvSoma'] = StateMonitor(n['Soma'],'V',record=True, clock=self.record_clock)
        mon['MiL'] = StateMonitor(n['Soma'],'iL',record=True, clock=self.record_clock)
        mon['MiSyn'] = StateMonitor(n['Soma'],'iSyn',record=True, clock=self.record_clock)
        mon['MiCoupleDend'] = StateMonitor(n['Soma'],'iCoupleDend',record=True, clock=self.record_clock)

    def make_model_soma_only(self, num_soma=1,clamped_current=0):
        '''
        Soma only, with current clamp
        :param num_soma:
        :param clamped_current: current injection (pA)
        :return:
        '''
        n = self.network
        p = self.params
        mon = self.monitor

        p['num_Soma'] = num_soma

        # Defining Neuron Group
        def myreset(P, spikes):
            P.V[spikes] = p['vReset_pyr']

        def myrefrac(P, spikes):
            P.V[spikes] = p['vReset_pyr']

        # Defining Neuron Group
        n['Soma'] = NeuronGroup(p['num_Soma'], Equations(self.eqs['Soma_only'], **p),
                                clock=self.simu_clock, threshold=p['vThres_pyr'],
                                reset=CustomRefractoriness(myreset,refracfunc=myrefrac,period=p['refracTime_pyr']))
        n['Soma'].V = p['vRest_pyr']
        n['Soma'].iClamp = clamped_current

        @network_operation(when='start',clock=self.simu_clock)
        def ClampedSoma():
            n['Soma'].iClamp = clamped_current

        self.contained_objects += [ClampedSoma]

        mon['MvSoma'] = StateMonitor(n['Soma'],'V',record=True, clock=self.record_clock)
        mon['MSpike'] = SpikeMonitor(n['Soma'])

    def activate_GABA_experiment(self,spiketime=200*ms):
        n = self.network
        c = self.connection
        p = self.params

        # Defining Neuron Group
        # Defining Inhibitory Synapses onto pyramidal dend
        n['InputDendGABA']=SpikeGeneratorGroup(p['num_Dend'],[(0,spiketime)],clock=self.simu_clock)
        c['DendGABA'] = IdentityConnection(n['InputDendGABA'],n['Dend'],'gTotGABA',weight = p['gGABA'])

    def spike_train_experiment(self, pre_time, post_times=None, num_input=1):
        '''
        Spike train experiment, specified by the exact spike trains of input and of the neuron
        :param pre_time: currently a scalar of pre-synaptic spike time
        :param post_times: array of post-synaptic spike time. When None, no post-synaptic spikes are forced.
        :param num_input: number of input synapses simultaneously activated
        :return:
        '''
        n = self.network
        c = self.connection
        p = self.params
        mon = self.monitor

        # Defining Neuron Group
        p['post_spike_time_list'] = post_times
        p['pre_EPSP_time'] = pre_time

        spiketimes = list()
        for i in xrange(num_input):
            spiketimes.append((i,p['pre_EPSP_time']))
        n['Input']=SpikeGeneratorGroup(num_input,spiketimes,clock=self.simu_clock)

        # AMPA synapses
        #c['AMPAsyn'] = Connection(n['Input'],n['Dend'][0],'gTotAMPA',weight=p['gAMPA'])

        # NMDA synapses and spine Calcium
        model_syn = '''
        dsNMDA/dt = -sNMDA/p['tauNMDADecay'] + sNMDARise*(1-sNMDA)*p['alphaNMDA'] : 1
        dsNMDARise/dt = -sNMDARise/p['tauNMDARise'] : 1
        dNMDACa/dt = (iNMDA/pamp-NMDACa)/p['tau_Ca_NMDA'] : 1
        iNMDA = -w*p['gNMDA']*sNMDA*(V_post-p['vE_pyr'])/(1+exp(-(V_post-p['vHalfNMDA'])/p['vSpreadNMDA'])) : pamp
        w : 1
        '''

        pre_code = 'sNMDARise = 1'
        c['NMDASyn']=Synapses(n['Input'],n['Dend'],
                            model=model_syn,
                            pre=pre_code,
                            clock=self.simu_clock_synapse)

        c['NMDASyn'][:,0] = True # Activating one dendrite
        c['NMDASyn'].w =1

        n['Dend'].iTotNMDA1 = c['NMDASyn'].iNMDA

        if post_times is not None:
            @network_operation(when='start',clock=self.simu_clock)
            def forcedPostSpike():
                if(any(abs(p['post_spike_time_list']-self.simu_clock.t)<self.simu_clock.dt/2)):
                    n['Soma'].V = p['vThres_pyr'] + 10*mV

            self.contained_objects += [forcedPostSpike]


        mon['MiNMDAsyn'] = StateMonitor(c['NMDASyn'],'iNMDA',record=True, clock=self.record_clock)
        mon['MNMDACasyn'] = StateMonitor(c['NMDASyn'],'NMDACa',record=True, clock=self.record_clock)
        mon['MbAPCa'] = StateMonitor(n['Soma'],'bAPCa',record=True, clock=self.record_clock)


    def rate_experiment(self, num_input=1, pre_rates = 0*Hz,
                        post_rate = 0*Hz, dend_inh_rate=0*Hz):
        '''
        Assume the dendrite-only model is used
        :param num_input: number of input onto each activated branch
        :param pre_rate: pre-synaptic rate
        :param post_rate: post-synaptic rate
        :param dend_inh_rate: dendritic inhibition rate

        Here the rate of the excitatory input is varied
        '''

        n = self.network
        c = self.connection
        p = self.params
        mon = self.monitor

        n_eachrate = num_input*p['num_Dend']//len(pre_rates)
        pre_rates_exp = repeat(pre_rates,n_eachrate)

        # Defining Neuron Group
        n['Input'] = PoissonGroup(num_input*p['num_Dend'],rates=pre_rates_exp,clock=self.simu_clock)

        # NMDA synapses and spine Calcium
        model_syn = '''
        dsNMDA/dt = -sNMDA/p['tauNMDADecay'] + sNMDARise*(1-sNMDA)*p['alphaNMDA'] : 1
        dsNMDARise/dt = -sNMDARise/p['tauNMDARise'] : 1
        dNMDACa/dt = (iNMDA/pamp-NMDACa)/p['tau_Ca_NMDA'] : 1
        iNMDA = -w*p['gNMDA']*sNMDA*(V_post-p['vE_pyr'])/(1+exp(-(V_post-p['vHalfNMDA'])/p['vSpreadNMDA'])) : pamp
        w : 1
        '''

        pre_code = 'sNMDARise = 1'
        c['NMDASyn']=Synapses(n['Input'],n['Dend'],
                            model=model_syn,
                            pre=pre_code,
                            clock=self.simu_clock_synapse)

        c['NMDASyn'][:,:] = "j==(i//num_input)" # The first num_input synapses connected to the first dendrite
        c['NMDASyn'].w =1 # gNMDA is set by p['gNMDA']

        n['Dend'].iTotNMDA1 = c['NMDASyn'].iNMDA


        # Defining Inhibitory Synapses onto pyramidal dend
        n['InputDendGABA'] = PoissonGroup(p['num_Dend'],rates=dend_inh_rate, clock=self.simu_clock)
        n['DendGABA'] = IdentityConnection(n['InputDendGABA'],n['Dend'],'gTotGABA',weight = p['gGABA'])

        # All dendrites receive the same post-synaptic spike trains
        n['ForcedSpike'] = PoissonGroup(1,rates=post_rate,clock=self.simu_clock)

        # Forcing post-synaptic spike
        c['bAPVoltageKick'] = Connection(n['ForcedSpike'],n['Dend'],'V',
                                              weight=p['bAP_amp'], delay=p['bAP_delay'])
        c['bAPCa'] = Connection(n['ForcedSpike'],n['Dend'],'bAPCa_rise',
                                              weight=p['norm_factor_Ca'])


        mon['MiNMDAsyn'] = StateMonitor(c['NMDASyn'],'iNMDA',record=True, clock=self.record_clock)
        mon['MNMDACasyn'] = StateMonitor(c['NMDASyn'],'NMDACa',record=True, clock=self.record_clock)
        mon['MbAPCa'] = StateMonitor(n['Dend'],'bAPCa',record=True, clock=self.record_clock)


    def weight_experiment(self, weights, num_input, pre_rate, dend_inh_rate,
                          post_rate=0):
        n = self.network
        c = self.connection
        p = self.params
        mon = self.monitor

        if p['num_Dend'] != p['num_Soma']:
            print p['num_Dend']
            print p['num_Soma']
            print 'each soma should have one dendrite for this experiment!!'
        n_weight = len(weights)
        n_rep = p['num_Soma']//n_weight # number of repetition for each value of synaptic weight

        # Defining Neuron Group

        n['Input'] = PoissonGroup(n_weight*n_rep*num_input,rates=pre_rate,clock=self.simu_clock)

        # NMDA synapses and spine Calcium
        model_syn = '''
        dsNMDA/dt = -sNMDA/p['tauNMDADecay'] + sNMDARise*(1-sNMDA)*p['alphaNMDA'] : 1
        dsNMDARise/dt = -sNMDARise/p['tauNMDARise'] : 1
        dNMDACa/dt = (iNMDA/pamp-NMDACa)/p['tau_Ca_NMDA'] : 1
        iNMDA = -w*p['gNMDA']*sNMDA*(V_post-p['vE_pyr'])/(1+exp(-(V_post-p['vHalfNMDA'])/p['vSpreadNMDA'])) : pamp
        w : 1
        '''

        pre_code = 'sNMDARise = 1'
        c['NMDASyn']=Synapses(n['Input'],n['Dend'],
                            model=model_syn,
                            pre=pre_code,
                            clock=self.simu_clock_synapse)

        c['NMDASyn'][:,:] = "i//num_input==j"
        weights = array(weights)
        weights_expand = weights.repeat(num_input*n_rep)
        c['NMDASyn'].w =weights_expand # gNMDA is set by p['gNMDA']


        n['Dend'].iTotNMDA1 = c['NMDASyn'].iNMDA


        # Defining Inhibitory Synapses onto pyramidal dend
        n['InputDendGABA'] = PoissonGroup(p['num_Dend'],rates=dend_inh_rate, clock=self.simu_clock)
        c['DendGABA'] = IdentityConnection(n['InputDendGABA'],n['Dend'],'gTotGABA',weight = p['gGABA'])

        # All dendrites receive the same post-synaptic spike trains
        n['ForcedSpike'] = PoissonGroup(1,rates=post_rate,clock=self.simu_clock)

        # Forcing post-synaptic spike
        c['bAPVoltageKick'] = Connection(n['ForcedSpike'],n['Dend'],'V',
                                              weight=p['bAP_amp'], delay=p['bAP_delay'])
        c['bAPCa'] = Connection(n['ForcedSpike'],n['Dend'],'bAPCa_rise',
                                              weight=p['norm_factor_Ca'])

        mon['MiNMDAsyn'] = StateMonitor(c['NMDASyn'],'iNMDA',record=True, clock=self.record_clock)
        mon['MNMDACasyn'] = StateMonitor(c['NMDASyn'],'NMDACa',record=True, clock=self.record_clock)
        mon['MbAPCa'] = StateMonitor(n['Dend'],'bAPCa',record=True, clock=self.record_clock)


    def single_pathway_gating_experiment(self, pre_rates, num_input=15, w_input = 1,record_EI=False):
        '''
        Single pathway gating experiment

        The neuron has multiple branches. And there are one pathway, which targets
        only one of the dendrite
        :param num_input: number of input per pathway onto each branch
        :param pre_rates: list contains input rates for each neuron
        :param w_input: scalar, input weight
        :return:
        '''
        n = self.network
        c = self.connection
        p = self.params

        # Defining Neuron Group
        pre_rates_path1 = repeat(pre_rates,num_input)
        n['Input_path1'] = PoissonGroup(num_input*p['num_Soma'],rates=pre_rates_path1,clock=self.simu_clock)

        # NMDA synapses and spine Calcium
        model_syn = '''
        dsNMDA/dt = -sNMDA/p['tauNMDADecay'] + sNMDARise*(1-sNMDA)*p['alphaNMDA'] : 1
        dsNMDARise/dt = -sNMDARise/p['tauNMDARise'] : 1
        iNMDA = -w*p['gNMDA']*sNMDA*(V_post-p['vE_pyr'])/(1+exp(-(V_post-p['vHalfNMDA'])/p['vSpreadNMDA'])) : pamp
        w : 1
        '''

        pre_code = 'sNMDARise = 1'

        # These work for neuron with two dendrites
        c['NMDASyn_path1']=Synapses(n['Input_path1'],n['Dend'],
                            model=model_syn,
                            pre=pre_code,
                            clock=self.simu_clock_synapse)
        # connecting to the first dendrite of every neuron
        c['NMDASyn_path1'][:,:] = "j==(i//num_input)*p['num_DendEach']"
        c['NMDASyn_path1'].w = [w_input]*(num_input*p['num_Soma'])

        n['Dend'].iTotNMDA1 = c['NMDASyn_path1'].iNMDA

        if record_EI:
            self.monitor['MgTotGABA'] = StateMonitor(n['Dend'],'gTotGABA',record=True, clock=self.record_clock)
            self.monitor['MsNMDA'] = StateMonitor(c['NMDASyn_path1'],'sNMDA',record=True, clock=self.record_clock)


    def single_pathway_gating_experiment_AMPA(self, pre_rates, num_input=15):
        '''
        Single pathway gating experiment

        The neuron has multiple branches. And there are one pathway, which targets
        only one of the dendrite
        :param pre_rates: list contains input rates for each neuron
        :param num_input: number of input per pathway onto each branch
        :return:
        '''
        n = self.network
        c = self.connection
        p = self.params


        pre_rates_path = zeros(p['num_Dend'])
        pre_rates_path[range(0,p['num_Dend'],p['num_DendEach'])]=pre_rates
        pre_rates_path = pre_rates_path*num_input

        n['Input_path1'] = PoissonGroup(p['num_Dend'],rates=pre_rates_path,clock=self.simu_clock)

        # Defining AMPA Synapses onto pyramidal dend
        c['AMPASyn_path1'] = IdentityConnection(n['Input_path1'],n['Dend'],'gTotAMPA',
                                                weight = p['gNMDA']) # set the weight to equal to NMDA conductance


    def single_pathway_gating_experiment_NMDAnonsatur(self, pre_rates, num_input=15, w_input = 1):
        '''
        Single pathway gating experiment with non-saturating NMDA channels


        The neuron has multiple branches. And there are one pathway, which targets
        only one of the dendrite
        :param num_input: number of input per pathway onto each branch
        :param pre_rates: list contains input rates for each neuron
        :param w_input: scalar, input weight
        :return:
        '''
        n = self.network
        c = self.connection
        p = self.params

        # Defining Neuron Group
        pre_rates_path1 = repeat(pre_rates,num_input)
        n['Input_path1'] = PoissonGroup(num_input*p['num_Soma'],rates=pre_rates_path1,clock=self.simu_clock)

        k = p['tauNMDARise']/p['tauNMDADecay']
        l = k**(-1/(1-1/k))-k**(-1/(k-1))

        # NMDA synapses and spine Calcium
        model_syn = '''
        dsNMDA/dt = -sNMDA/p['tauNMDADecay']+sNMDARise/p['tauNMDARise'] : 1
        dsNMDARise/dt = -sNMDARise/p['tauNMDARise'] : 1
        iNMDA = -w*p['gNMDA']*sNMDA/l*(V_post-p['vE_pyr'])/(1+exp(-(V_post-p['vHalfNMDA'])/p['vSpreadNMDA'])) : pamp
        w : 1
        '''

        pre_code = 'sNMDARise = 1'

        # These work for neuron with two dendrites
        c['NMDASyn_path1']=Synapses(n['Input_path1'],n['Dend'],
                            model=model_syn,
                            pre=pre_code,
                            clock=self.simu_clock_synapse)
        # connecting to the first dendrite of every neuron
        c['NMDASyn_path1'][:,:] = "j==(i//num_input)*p['num_DendEach']"
        c['NMDASyn_path1'].w = [w_input]*(num_input*p['num_Soma'])

        n['Dend'].iTotNMDA1 = c['NMDASyn_path1'].iNMDA

    def two_pathway_gating_experiment_backup(self, pre_rates, num_input=15, dend_inh_rate=100*Hz, w_input = (1,1), NMDA_prop = 1):
        '''
        Basic two-pathway gating experiment

        :param num_input: number of input per pathway onto each branch
        :param pre_rates: list of two lists, each sub-list contains input rates for one pathway
        :param dend_inh_rate: inhibition rate for inhibited dendrite
        :param w_input: pair of weight, first element high weight, second element low weight
        :return:
        '''
        n = self.network
        c = self.connection
        p = self.params

        n_tar = 2 # number of dendrites targeted by one pathway

        # Defining Neuron Group
        if NMDA_prop>0:
            # NMDA synapses
            pre_rates_path1 = repeat(pre_rates[0],num_input)
            pre_rates_path2 = repeat(pre_rates[1],num_input)
            n['Input_path1'] = PoissonGroup(num_input*p['num_Soma'],rates=pre_rates_path1,clock=self.simu_clock)
            n['Input_path2'] = PoissonGroup(num_input*p['num_Soma'],rates=pre_rates_path2,clock=self.simu_clock)


            model_syn = '''
            dsNMDA/dt = -sNMDA/p['tauNMDADecay'] + sNMDARise*(1-sNMDA)*p['alphaNMDA'] : 1
            dsNMDARise/dt = -sNMDARise/p['tauNMDARise'] : 1
            iNMDA = -w*p['gNMDA']*sNMDA*(V_post-p['vE_pyr'])/(1+exp(-(V_post-p['vHalfNMDA'])/p['vSpreadNMDA'])) : pamp
            w : 1
            '''

            pre_code = 'sNMDARise = 1'

            c['NMDASyn_path1']=Synapses(n['Input_path1'],n['Dend'],
                                model=model_syn,
                                pre=pre_code,
                                clock=self.simu_clock_synapse)
            # Here target the first n_path*n_tar dendrites
            c['NMDASyn_path1'][:,:] = "((i//num_input)*p['num_DendEach']<=j)&((i//num_input)*p['num_DendEach']+2*n_tar-1>=j)"
            # Weight on to each of the four dendrites
            w_path1 = [w_input[0]]*n_tar+[w_input[1]]*n_tar
            w_path1 = w_path1*(num_input*p['num_Soma'])
            w_path1 = array(w_path1)*NMDA_prop
            c['NMDASyn_path1'].w =w_path1


            c['NMDASyn_path2']=Synapses(n['Input_path2'],n['Dend'],
                                model=model_syn,
                                pre=pre_code,
                                clock=self.simu_clock_synapse)
            c['NMDASyn_path2'][:,:] = "((i//num_input)*p['num_DendEach']<=j)&((i//num_input)*p['num_DendEach']+2*n_tar-1>=j)"
            w_path2 = [w_input[1]]*n_tar+[w_input[0]]*n_tar
            w_path2 = w_path2*(num_input*p['num_Soma'])
            w_path2 = array(w_path2)*NMDA_prop
            c['NMDASyn_path2'].w =w_path2


            n['Dend'].iTotNMDA1 = c['NMDASyn_path1'].iNMDA
            n['Dend'].iTotNMDA2 = c['NMDASyn_path2'].iNMDA

        if NMDA_prop<1:
            # Defining AMPA Synapses onto pyramidal dend
            pre_rates_AMPA_path_w0_exp = [[pre_rates[0][i]]*n_tar+[pre_rates[1][i]]*n_tar+[0]*(p['num_DendEach']-2*n_tar) for i in xrange(p['num_Soma'])]
            pre_rates_AMPA_path_w0_exp = array(pre_rates_AMPA_path_w0_exp).flatten()*num_input
            n['Input_AMPA_w0'] = PoissonGroup(p['num_Dend'],rates=pre_rates_AMPA_path_w0_exp, clock=self.simu_clock)
            c['AMPA_w0'] = IdentityConnection(n['Input_AMPA_w0'],n['Dend'],'gTotAMPA',weight = w_input[0]*p['gNMDA']*(1-NMDA_prop))

            pre_rates_AMPA_path_w1_exp = [[pre_rates[1][i]]*n_tar+[pre_rates[0][i]]*n_tar+[0]*(p['num_DendEach']-2*n_tar) for i in xrange(p['num_Soma'])]
            pre_rates_AMPA_path_w1_exp = array(pre_rates_AMPA_path_w1_exp).flatten()*num_input
            n['Input_AMPA_w1'] = PoissonGroup(p['num_Dend'],rates=pre_rates_AMPA_path_w1_exp, clock=self.simu_clock)
            c['AMPA_w1'] = IdentityConnection(n['Input_AMPA_w1'],n['Dend'],'gTotAMPA',weight = w_input[1]*p['gNMDA']*(1-NMDA_prop))

        # Defining Inhibitory Synapses onto pyramidal dend
        dend_inh_rates_exp = ([0]*n_tar+[dend_inh_rate]*n_tar+[0]*(p['num_DendEach']-2*n_tar))*p['num_Soma']
        n['InputDendGABA'] = PoissonGroup(p['num_Dend'],rates=dend_inh_rates_exp, clock=self.simu_clock)
        c['DendGABA'] = IdentityConnection(n['InputDendGABA'],n['Dend'],'gTotGABA',weight = p['gGABA'])

    def two_pathway_gating_experiment(self, pre_rates, num_input=15,
                                      dend_inh_rate=100*Hz, w_input = (1,1),
                                      NMDA_prop = 1,GABAA_prop = 1):
        '''
        Basic two-pathway gating experiment

        :param num_input: number of input per pathway onto each branch
        :param pre_rates: list of two lists, each sub-list contains input rates for one pathway
        :param dend_inh_rate: inhibition rate for inhibited dendrite
        :param w_input: pair of weight, first element high weight, second element low weight
        :return:
        '''
        n = self.network
        c = self.connection
        p = self.params

        n_tar = 2 # number of dendrites targeted by one pathway

        # Defining Neuron Group
        if NMDA_prop>0:
            # NMDA synapses
            pre_rates_path1 = repeat(pre_rates[0],num_input)
            pre_rates_path2 = repeat(pre_rates[1],num_input)
            n['Input_path1'] = PoissonGroup(num_input*p['num_Soma'],rates=pre_rates_path1,clock=self.simu_clock)
            n['Input_path2'] = PoissonGroup(num_input*p['num_Soma'],rates=pre_rates_path2,clock=self.simu_clock)


            model_syn = '''
            dsNMDA/dt = -sNMDA/p['tauNMDADecay'] + sNMDARise*(1-sNMDA)*p['alphaNMDA'] : 1
            dsNMDARise/dt = -sNMDARise/p['tauNMDARise'] : 1
            iNMDA = -w*p['gNMDA']*sNMDA*(V_post-p['vE_pyr'])/(1+exp(-(V_post-p['vHalfNMDA'])/p['vSpreadNMDA'])) : pamp
            w : 1
            '''

            pre_code = 'sNMDARise = 1'

            c['NMDASyn_path1']=Synapses(n['Input_path1'],n['Dend'],
                                model=model_syn,
                                pre=pre_code,
                                clock=self.simu_clock_synapse)
            # Here target the first n_path*n_tar dendrites
            c['NMDASyn_path1'][:,:] = "((i//num_input)*p['num_DendEach']<=j)&((i//num_input)*p['num_DendEach']+2*n_tar-1>=j)"
            # Weight on to each of the four dendrites
            w_path1 = [w_input[0]]*n_tar+[w_input[1]]*n_tar
            w_path1 = w_path1*(num_input*p['num_Soma'])
            w_path1 = array(w_path1)*NMDA_prop
            c['NMDASyn_path1'].w =w_path1


            c['NMDASyn_path2']=Synapses(n['Input_path2'],n['Dend'],
                                model=model_syn,
                                pre=pre_code,
                                clock=self.simu_clock_synapse)
            c['NMDASyn_path2'][:,:] = "((i//num_input)*p['num_DendEach']<=j)&((i//num_input)*p['num_DendEach']+2*n_tar-1>=j)"
            w_path2 = [w_input[1]]*n_tar+[w_input[0]]*n_tar
            w_path2 = w_path2*(num_input*p['num_Soma'])
            w_path2 = array(w_path2)*NMDA_prop
            c['NMDASyn_path2'].w =w_path2


            n['Dend'].iTotNMDA1 = c['NMDASyn_path1'].iNMDA
            n['Dend'].iTotNMDA2 = c['NMDASyn_path2'].iNMDA

        if NMDA_prop<1:
            # Defining AMPA Synapses onto pyramidal dend
            pre_rates_AMPA_path_w0_exp = [[pre_rates[0][i]]*n_tar+[pre_rates[1][i]]*n_tar+[0]*(p['num_DendEach']-2*n_tar) for i in xrange(p['num_Soma'])]
            pre_rates_AMPA_path_w0_exp = array(pre_rates_AMPA_path_w0_exp).flatten()*num_input
            n['Input_AMPA_w0'] = PoissonGroup(p['num_Dend'],rates=pre_rates_AMPA_path_w0_exp, clock=self.simu_clock)
            c['AMPA_w0'] = IdentityConnection(n['Input_AMPA_w0'],n['Dend'],'gTotAMPA',weight = w_input[0]*p['gNMDA']*(1-NMDA_prop))

            pre_rates_AMPA_path_w1_exp = [[pre_rates[1][i]]*n_tar+[pre_rates[0][i]]*n_tar+[0]*(p['num_DendEach']-2*n_tar) for i in xrange(p['num_Soma'])]
            pre_rates_AMPA_path_w1_exp = array(pre_rates_AMPA_path_w1_exp).flatten()*num_input
            n['Input_AMPA_w1'] = PoissonGroup(p['num_Dend'],rates=pre_rates_AMPA_path_w1_exp, clock=self.simu_clock)
            c['AMPA_w1'] = IdentityConnection(n['Input_AMPA_w1'],n['Dend'],'gTotAMPA',weight = w_input[1]*p['gNMDA']*(1-NMDA_prop))

        # Defining Inhibitory Synapses onto pyramidal dend
        dend_inh_rates_exp = ([0]*n_tar+[dend_inh_rate]*n_tar+[0]*(p['num_DendEach']-2*n_tar))*p['num_Soma']
        n['InputDendGABA'] = PoissonGroup(p['num_Dend'],rates=dend_inh_rates_exp, clock=self.simu_clock)
        if GABAA_prop>0:
            c['DendGABA'] = IdentityConnection(n['InputDendGABA'],n['Dend'],'gTotGABA',weight = p['gGABA']*GABAA_prop)
        if GABAA_prop<1:
            c['DendGABAB'] = IdentityConnection(n['InputDendGABA'],n['Dend'],'gTotGABABRise',weight = p['gGABA']*(1-GABAA_prop))

    def make_invivo_bkginput(self):
        p = self.params
        n = self.network
        # in vivo condition possibilities
        vivo = dict()
        vivo['bgE'] = PoissonGroup(p['num_Soma'],rates=p['rate_bgE'],clock=self.simu_clock)
        vivo['bgI'] = PoissonGroup(p['num_Soma'],rates=p['rate_bgI'],clock=self.simu_clock)
        vivo['backgroundE'] = IdentityConnection(vivo['bgE'], n['Soma'], 'gTotAMPA',weight=p['weight_bgE'])
        vivo['backgroundI'] = IdentityConnection(vivo['bgI'], n['Soma'], 'gTotGABA',weight=p['weight_bgI'])
        self.contained_objects += vivo.values()


    def make_dend_bkginh(self, inh_rates=0*Hz):
        p = self.params
        n = self.network
        c = self.connection
        n['dend_bkginh'] = PoissonGroup(p['num_Dend'],rates=inh_rates,clock=self.simu_clock)
        c['dend_bkginh'] = IdentityConnection(n['dend_bkginh'], n['Dend'], 'gTotGABA',weight=p['gGABA'])


    def make_network(self):
        self.contained_objects += self.network.values()
        self.contained_objects += self.connection.values()
        self.contained_objects += self.monitor.values()


    def reinit(self):
        n = self.network
        p = self.params
        mon = self.monitor

        # Reset all network components
        for g in n.values():
            g.reinit()
        for g in mon.values():
            g.reinit()

        # Reset membrane potential

        # Initialization
        if 'Soma' in n.keys():
            n['Soma'].V = p['vRest_pyr']
            n['Soma'].vShadow = p['vRest_pyr']
            n['Soma'].vAveDend = p['vRest_pyr']

        if 'Dend' in n.keys():
            n['Dend'].V = p['vRest_pyr']
            n['Dend'].vSoma = p['vRest_pyr']

        # Reset
        self.simu_clock.reinit()
        self.simu_clock_synapse.reinit()
        self.record_clock.reinit()

        print 'Network Reset'


    def get_CaTrace(self):
        c = self.connection
        p = self.params
        mon = self.monitor

        NMDACasyn = mon['MNMDACasyn'].values
        num_syn = NMDACasyn.shape[0]

        bAPCa = mon['MbAPCa'].values
        num_soma = bAPCa.shape[0]
        synsomaratio = num_syn//num_soma

        bAPCasyn = bAPCa.repeat(synsomaratio,axis=0)

        CaTrace = NMDACasyn*p['NMDA_scaling'] + bAPCasyn*p['bAP_scaling']
        return CaTrace


    def update_weight(self, CaTrace=None, repeat_times=1, real_update=True):
        '''
        update weight with Ca-based plasticity
        :param T: total time of simulation
        :param repeat_times: number of repeat sessions
        :return: nothing
        '''
        c = self.connection
        p = self.params
        mon = self.monitor
        if CaTrace is None:
            CaTrace = self.get_CaTrace()

        num_syn = len(c['NMDASyn'].w[:])
        wpost_list = zeros(num_syn)

        for i_syn in xrange(num_syn):
            CaTraceSorted = sort(CaTrace[i_syn])

            AlphaP = crossthresholdtime(CaTraceSorted,p['thetaP'])*p['record_dt']*repeat_times
            AlphaD = crossthresholdtime(CaTraceSorted,p['thetaD'])*p['record_dt']*repeat_times

            wpre = c['NMDASyn'].w[i_syn]
            wpost = SynapseChange(wpre,AlphaP,AlphaD,p)
            wpost_list[i_syn] = wpost
        if real_update:
            c['NMDASyn'].w[:] = wpost_list

        return wpost_list

