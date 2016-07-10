# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:23:24 2015

@author: guangyuyang

This file should run the simulations for every figure in Yang, Murray, Wang (2015)
"""

from __future__ import division

figpath = 'figure/'
datapath = 'data/'
version = ''

args = {'figpath':figpath,
        'datapath':datapath,
        'version':version,
        'paramsfile':datapath+'parameters.txt',
        'eqsfile':datapath+'equations.txt'}

#=============================================================================
# Pathway-specific gating
#=============================================================================
import PathwayGating
PG = PathwayGating.Study(**args)

# Run the neuron with two pathways of input, with varying strength of each pathway (different modes)
#PG.run_basic_pathway_gating(mode='specific')
#PG.plot_basic_pathway_gating(mode='specific')

#PG.run_basic_pathway_gating(mode='specific_closed')
#PG.plot_basic_pathway_gating(mode='specific_closed')

#PG.run_basic_pathway_gating(mode='specific',run_slice=True)
#PG.plot_basic_pathway_gating_slice(mode='specific')

#PG.run_basic_pathway_gating(mode='AMPA')
#PG.plot_basic_pathway_gating(mode='AMPA')


#=============================================================================
# Fitting the rate model of pyramidal neurons
#=============================================================================
import RateModelFitting

#RMF = RateModelFitting.Study(**args)

# Fit dendritic voltage as a function of excitatory weight and inhibitory rate
#RMF.fit_DendVvsgErI()
# Somatic firing as a function of dendritic voltage
#RMF.fit_RatevsDendV()
# Somatic firing as a function of current injection
#RMF.fit_RatevsI()


#=============================================================================
# Gating selectivity in single neuron
#=============================================================================
import GatingSelectivity

GS = GatingSelectivity.Study(**args)
# Vary number of disinhibited dendrites
#GS.plot_vary_n_disinh(savenew=True,plot_ylabel=True)
# Vary strength of inhibition
#GS.plot_vary_inh(savenew=True,plot_ylabel=False)
# Vary AMPA NMDA ratio
#GS.run_NMDAAMPAratio()
#GS.plot_NMDAAMPAratio()
# Vary GABAA, GABAB ratio
#GS.run_GABAABratio()
#GS.plot_GABAABratio()
# Vary NMDA and GABAA ratio
#GS.run_NMDAAMPAGABAABratio()
#GS.plot_NMDAAMPAGABAABratio()

#=============================================================================
# Gating selectivity in interneuronal circuit
#=============================================================================
import InterneuronalCircuit

IC = InterneuronalCircuit.Study(**args)
#IC.run_vary_x(plot_type_list=['n_som2dend','n_dend_each','n_som','p_som2pyr'],N_rnd=10,modeltype='SOM_alone')
#IC.plot_vary_x('n_som2dend',set_yticklabels=False,modeltype='SOM_alone')
#IC.plot_vary_x('n_dend_each',set_yticklabels=True,modeltype='SOM_alone')
#IC.plot_vary_x('n_som',set_yticklabels=False,modeltype='SOM_alone')
#IC.plot_vary_x('p_som2pyr',set_yticklabels=True,modeltype='SOM_alone')
                                     
#IC.run_vary_x(plot_type_list=['p_vip2som','p_exc2vip','p_exc2som'],N_rnd=10,modeltype='Control2VIPSOM')
#IC.plot_vary_p_exc2neurons()
#IC.plot_vary_x('p_vip2som',set_yticklabels=False,modeltype='Control2VIPSOM')

#IC.run_vary_x(plot_type_list=['p_vip2som'],N_rnd=20,modeltype='Control2VIP')
#IC.plot_vary_x('p_exc2vip',set_yticklabels=True,modeltype='Control2VIP')
#IC.plot_vary_x('p_vip2som',set_yticklabels=False,modeltype='Control2VIP')

# Plot input to SOM neurons
#IC.plot_inputs(modeltype='Control2VIPSOM')
#IC.plot_inputs(modeltype='Control2VIP')

# Vary somatic inhibition
#IC.run_fancyvary_somainh(N_rnd=10)
#IC.plot_fancyvary_somainh()

#=============================================================================
# Newsome Task
#=============================================================================
import NewsomeTask
#args2 = args.copy()
#args2['seed'] = 4
#NT = NewsomeTask.Study(**args2)
#NT.fit_run_model(monkey='F')


#=============================================================================
# Disinhibition regulating synaptic plasticity
#=============================================================================
import DisinhibitionRegulatePlasticity
#DRP = DisinhibitionRegulatePlasticity.Study(**args)

# Calcium traces
#DRP.compareWithNevian_CaTrace()

# How weight change depends on excitatory input rate and inhibition
#DRP.run_WeightChangevsRate()
#DRP.calc_WeightChangevsRate()
#DRP.plot_WeightChangevsRate()

# How weight change depends on initial weight
#DRP.run_WeightChangevsInitWeight()
##DRP.plot_WeightChangevsInitWeight()

# Run various tuning curves
#PG.run_basic_pathway_gating(mode='prelearning')
#PG.plot_basic_pathway_gating(mode='prelearning',plot_ylabel=False)
#PG.run_basic_pathway_gating(mode='postlearning')
#PG.plot_basic_pathway_gating(mode='postlearning')

#PG.run_basic_pathway_gating(mode='prelearning',run_slice=True)
#PG.plot_basic_pathway_gating_slice(mode='prelearning')
#PG.run_basic_pathway_gating(mode='postlearning',run_slice=True)
#PG.plot_basic_pathway_gating_slice(mode='postlearning',plot_ylabel=False)


#os.system('say "your program has finished"')
