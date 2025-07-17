'''
Polymer simulation with dynamic spreading of chromatin state.  The simulation alters between cycles of 3D brownian polymer motion and epigenetic spreading.
The epigenetic state of the monomers affects their interaction affinity with one another during the 3D simulation. The 3D proximities at the end of each cycle of 3D motion affect the epigenetic spreading. 

The simulations have been run with either 3 Pc-monomer states or a single state, as described below.
Additional details about the simulation are including in our accompanying Text and Methods. 

This example runs with 3 levels of Pc states (as in Figure 7) and an adhesivity of 0.4.  These values may be updated as described below.  

To execute the script, run `python PcSim_3PcState_Example.py "Path\To\Save_Folder"`
where Path\To\Save_Folder is the filepath in which to save the simulation results. 

The simulation requires the BoettigerLab fork of the open2c/polychrom package, available here: https://github.com/BoettigerLab/polychrom
This package uses the OpenMM, GPU accelerated framework for molecular dynamics simulations. Additional guidance on install and configuration are available here:  http://docs.openmm.org/7.0.0/userguide/application.html 

'''

# import some libaries

import sys
import os
import numpy as np
import numpy.matlib
import h5py
# import json
import pandas as pd
import math
import copy
# # Python's continual incompetence with finding packages
sys.path.append("C:\Shared\polychrom-shared")  # even the script version needs help 
sys.path.append("C:\Shared\polychrom-shared\Simulations") # only needed in jupyter notebook. 

from LEBondUpdater import bondUpdater
import polychrom
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
from polychrom.simulation import Simulation
from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
import time
# added
import scipy
from scipy import spatial  
import pickle # for saving parameters 
import ast

#===========================general parameters ========================================
#  We call this function from a batch manager which passes it a target save folder and an iteration index for the parameter scan.  
#    The first step after the imports is to parse these values and create the save folder 

topFolder = sys.argv[1] # 
sticky = 0.4 # Adhesive interaction between Pc-monomers. Values of ~.7 are sufficient in either the 1-state or 3-state model to create a collapsed globular 'droplet' structure.  
saveFolder = topFolder + 'sim001' + '/'


if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)  # only creates folders 1 deep, won't create a full path
    

# ---------------- color dynamic pars
iters = 2000 # number of iterations to do  
totPc = 1/3 # Enough Pc to cover at most 1/3rd of the domain.  
onRate = 0.25 #   Catalyzed rate of monomer state transition from state n -> n+1, for n=[0..2]  in the 3-state model, or just n=0 in the 1-state model
onRateBkd = 0.0015;  # spontaneous rate of monomer state transition from state n -> n+1 for n=[0..2] " " "  
offRate = 0.05 #   spontaneous rate of monomer state tranition from state n -> n-1 for n= [1..3] in the 3-state model or just for n=1 in the 1-state model
contactRadius = 1.75 # contact radius (in units of the hard-sphere monomer radius, at which state transitions can be catalyzed
nStates = 3 # Total number of Pc-monomer states in the model.  1 or 3 in our simulations
startLevel = 2 # Starting state for the Pc-monomers (1-3 produce qualitatively similar results, as the catalytic addition quickly tops off the intitial states.  Starting at 0 produces completely different results, as there in this case both the "pc-region" and non-pc region start in the same state (0) and there is no memory to be propigated.  The spontaneous additions results in random seeding. This is condition is outside of the case that motivated by the epigenetics of the hox domain. 
 
#-------------------- Sticky polymer dynamics 
N = 600 #  Number of monomers in the polymer
density =  0.2  # density of the periodic boundary condition box. 
attraction_radius = 1.5  #  distance (relative to hard-sphere monomer radius) at which monomers stick to eachother in the Lenard-Jones potential.
num_chains = 1  # number of separate chains in the simulation (they will stick together)
oneChainMonomerTypes = np.zeros(N).astype(int)
oneChainMonomerTypes[200:400] = 1 # mod self interaction
# create interaction matrix
interactionMatrix = np.array([[0, 0], [0, sticky]])  #   0.8  # === KEY parameter  ===#  
# ==== coil globule transition occurs around .7 for monomers of this length


#  ------------ Extrusion sim parameters   
#  with LEFNum = 0 corresponds to a case with no cohesin, in which case the positions of the CTCF LEF barriers become irrelevant
#  These parameters are retained here to facilitate future explorations of the effect of cohesin loop extrusion and cohesin-CTCF dynamics on polycomb spreading.
#  The code-framework to for the loop-extrusion simulations is likewise retained for this purpose.  For understanding the simulations in our current manuscript these may be safely ignored.  These additional 1D simulations create a negligible addition to the total run time of the simulations.  
LEFNum =  0 #
MDstepsPerCohesinStep = 800
smcBondWiggleDist = 0.2
smcBondDist = 0.5
LIFETIME =  50 #  [Imakaev/Mirny use 200 as demo] extruder lifetime
ctcfSites =  np.array([200,400]) #  np.array([0,399,400]) #CTCF site locations  # positioned on HoxA
nCTCF = np.shape(ctcfSites)[0]
ctcfDir = np.zeros(nCTCF) # 0 is bidirectional, 1 is right 2 is left
ctcfCapture = 0.99*np.ones(nCTCF) #  capture probability per block if capture < than this, capture  
ctcfRelease =0.003*np.ones(nCTCF)  # % release probability per block. if capture < than this, release
loadProb = np.ones([1,N])  # uniform loading probability
loadProb = numpy.matlib.repmat(loadProb,1,1) # need to replicate and renormalize
loadProb = loadProb/np.sum(loadProb) 
lefPosFile = saveFolder + "LEFPos.h5"
nCTCF = np.shape(ctcfSites)[0]
saveEveryBlocks = 10   # save every 10 blocks 
restartSimulationEveryBlocks = 100  # blocks per iteration
trajectoryLength =  iters*restartSimulationEveryBlocks #  1000 # time duration of simulation (down from 100,000)
monomers = N



#-------------- Run Cohesin simulation if required
#  These parameters are retained here to facilitate future explorations of the effect of cohesin loop extrusion and cohesin-CTCF dynamics on polycomb spreading.
#  In order to maintain generality in the coding framework, some parameters will be iniatilized in this section that are referenced in the 3D simulation.
import polychrom.lib.extrusion1Dv2 as ex1D # 1D classes 
ctcfLeftRelease = {}
ctcfRightRelease = {}
ctcfLeftCapture = {}
ctcfRightCapture = {}
for i in range(num_chains): # loop over chains
    for t in range(len(ctcfSites)):
        pos = i * N + ctcfSites[t] 
        if ctcfDir[t] == 0:
            ctcfLeftCapture[pos] = ctcfCapture[t]  # if random [0,1] is less than this, capture
            ctcfLeftRelease[pos] = ctcfRelease[t]  # if random [0,1] is less than this, release
            ctcfRightCapture[pos] = ctcfCapture[t]
            ctcfRightRelease[pos] = ctcfRelease[t]
        elif ctcfDir[t] == 1: # stop Cohesin moving toward the right  
            ctcfLeftCapture[pos] = 0  
            ctcfLeftRelease[pos] = 1  
            ctcfRightCapture[pos] = ctcfCapture[t]
            ctcfRightRelease[pos] = ctcfRelease[t]
        elif ctcfDir[t] == 2:
            ctcfLeftCapture[pos] = ctcfCapture[t]  # if random [0,1] is less than this, capture
            ctcfLeftRelease[pos] = ctcfRelease[t]  # if random [0,1] is less than this, release
            ctcfRightCapture[pos] = 0
            ctcfRightRelease[pos] = 1
       
args = {}
args["ctcfRelease"] = {-1:ctcfLeftRelease, 1:ctcfRightRelease}
args["ctcfCapture"] = {-1:ctcfLeftCapture, 1:ctcfRightCapture}        
args["N"] = N 
args["LIFETIME"] = LIFETIME
args["LIFETIME_STALLED"] = LIFETIME  # no change in lifetime when stalled 

occupied = np.zeros(N)
occupied[0] = 1  # (I think this is just prevent the cohesin loading at the end by making it already occupied)
occupied[-1] = 1 # [-1] is "python" for end
cohesins = []

print('starting simulation with N LEFs=')
print(LEFNum)
for i in range(LEFNum):
    ex1D.loadOneFromDist(cohesins,occupied, args,loadProb) # load the cohesins 


with h5py.File(lefPosFile, mode='w') as myfile:    
    dset = myfile.create_dataset("positions", 
                                 shape=(trajectoryLength, LEFNum, 2), 
                                 dtype=np.int32, 
                                 compression="gzip")
    steps = 100    # saving in 50 chunks because the whole trajectory may be large 
    bins = np.linspace(0, trajectoryLength, steps, dtype=int) # chunks boundaries 
    for st,end in zip(bins[:-1], bins[1:]):
        cur = []
        for i in range(st, end):
            ex1D.translocate(cohesins, occupied, args,loadProb)  # actual step of LEF dynamics 
            positions = [(cohesin.left.pos, cohesin.right.pos) for cohesin in cohesins]
            cur.append(positions)  # appending current positions to an array 
        cur = np.array(cur)  # when we finished a block of positions, save it to HDF5 
        dset[st:end] = cur
    myfile.attrs["N"] = N
    myfile.attrs["LEFNum"] = LEFNum
trajectory_file = h5py.File(lefPosFile, mode='r')
LEFNum = trajectory_file.attrs["LEFNum"]  # number of LEFs
LEFpositions = trajectory_file["positions"]  # array of LEF positions  
steps = MDstepsPerCohesinStep # MD steps per step of cohesin  (set to ~800 in real sims)
Nframes = LEFpositions.shape[0] # length of the saved trajectory (>25000 in real sims)
block = 0  # starting block 
# test some properties 
# assertions for easy managing code below 
assert (Nframes % restartSimulationEveryBlocks) == 0 
assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0
savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
simInitsTotal  = (Nframes) // restartSimulationEveryBlocks
if len(oneChainMonomerTypes) != N: # copy monomer states to all chains if multiple chains are used
    monomerTypes = np.tile(oneChainMonomerTypes, num_chains)
else:
    monomerTypes = oneChainMonomerTypes   
N_chain = len(oneChainMonomerTypes)  
N = len(monomerTypes)
print(f'N_chain: {N_chain}')  # ~8000 in a real sim
print(f'N: {N}')   # ~40000 in a real sim
N_traj = trajectory_file.attrs["N"]
print(f'N_traj: {N_traj}')
assert N == trajectory_file.attrs["N"]
print(f'Nframes: {Nframes}')
print(f'simInitsTotal: {simInitsTotal}')



#==============================================================#
#                  RUN 3D simulation                              #
#==============================================================#
import shutil
# Initial simulation using fixed input states
t=0
LEFsubset = LEFpositions[t*restartSimulationEveryBlocks:(t+1)*restartSimulationEveryBlocks,:,:] # a subset of the total LEF simulation time
milker = bondUpdater(LEFsubset)
data = grow_cubic(N,int((N/(density*1.2))**0.333))  # starting conformation
PBC_width = (N/density)**0.333
chains = [(N_chain*(k),N_chain*(k+1),0) for k in range(num_chains)]  # subchains in rpt
newFolder = saveFolder+'t'+str(0)+'/'
if os.path.exists(newFolder):
    shutil.rmtree(newFolder)
os.mkdir(newFolder)
reporter = HDF5Reporter(folder=newFolder, max_data_length=100)
a = Simulation(N=N, 
               error_tol=0.01, 
               collision_rate=0.02, 
               integrator ="variableLangevin", 
               platform="cuda",   
               GPU = "0",   # 
               PBCbox=(PBC_width, PBC_width, PBC_width),
               reporters=[reporter],
               precision="mixed")  # platform="CPU", "cuda", "OpenCL"
a.set_data(data) # initial polymer 
a.add_force(
    polychrom.forcekits.polymer_chains(
        a,
        chains=chains,
        nonbonded_force_func=polychrom.forces.heteropolymer_SSW,
        nonbonded_force_kwargs={
            'attractionEnergy': 0,  # base attraction energy for all monomers
            'attractionRadius': attraction_radius,
            'interactionMatrix': interactionMatrix,
            'monomerTypes': monomerTypes,
            'extraHardParticlesIdxs': []
        },
        bond_force_kwargs={
            'bondLength': 1,
            'bondWiggleDistance': 0.05
        },
        angle_force_kwargs={
            'k': 1.5
        }
    )
)
# ------------ initializing milker; adding bonds ---------
kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
bondDist = smcBondDist * a.length_scale
activeParams = {"length":bondDist,"k":kbond}
inactiveParams = {"length":bondDist, "k":0}
milker.setParams(activeParams, inactiveParams)  
milker.setup(bondForce=a.force_dict['harmonic_bonds'],
            blocks=restartSimulationEveryBlocks)
# If your simulation does not start, consider using energy minimization below
a.local_energy_minimization()  # only do this at the beginning

# this runs 
for i in range(restartSimulationEveryBlocks):   # loops over 100     
    if i % saveEveryBlocks == (saveEveryBlocks - 1):  
        a.do_block(steps=steps)
    else:
        a.integrator.step(steps)  # do steps without getting the positions from the GPU (faster)
    if i < restartSimulationEveryBlocks - 1: 
        curBonds, pastBonds = milker.step(a.context)  # this updates bonds. You can do something with bonds here
data = a.get_data()  # save data and step, and delete the simulation
del a    
reporter.blocks_only = True  # Write output hdf5-files only for blocks
time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)
reporter.dump_data()

# -------------------------------end of initialization. ---------------------------
# ===============   Time to start updating the color states  ================================#  
colorStates =  np.tile(startLevel*monomerTypes, [iters+1,1]) # initialize a matrix to store color states in. 
for t in range(iters):
    print(t)
    #==================== simulate epigenetic spreading 
    # load polymer
    # files = list_URIs(saveFolder)
    # data = load_URI(files[-1])  # this is the full data structure, it is possible we only want data['pos']

    newFolder = saveFolder+'t'+str(t+1)+'/'
    if not os.path.exists(newFolder):
        os.makedirs(newFolder)
    reporter = HDF5Reporter(folder=newFolder, max_data_length=100)
    print('creating folder')

    for p in range(1):
        polyDat = data[p*monomers:(p+1)*monomers,:]  # ['pos'][p*monomers:(p+1)*monomers,:]
        newColors = copy.copy(colorStates[t,p*monomers:(p+1)*monomers]) # note this is not a copy, just a reference. updating newColors immideately updates colorStates
        # moved frac bound down
        isLoss = np.random.rand(monomers) < offRate
        newColors[isLoss] = newColors[isLoss]-1 # was 0  # note, this immideately updates colorStates
        newColors[newColors<0] = 0
        fracBound = sum(newColors)/monomers/nStates
        fracFree = max(0,totPc-fracBound)
        ordr = np.random.permutation(monomers)
        dMap = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(polyDat))
        isOn = newColors # newColors == 1
        for o in ordr:
            isClose = dMap[o,:] < contactRadius
            tries = sum(isClose * isOn)  
            updateProb = onRate*tries*fracFree
            updateColor = np.random.rand() < updateProb  or  (np.random.rand(1) < onRateBkd*fracFree)  # ADDED FOR background integration
            if updateColor and newColors[o]<nStates:
                newColors[o] = newColors[o] + 1  # note, this immideately updates colorStates  
        colorStates[t+1,p*monomers:(p+1)*monomers] = newColors
    # ============ run new polymer sim  ==========================
    
    isPc =  colorStates[t+1,:]>0
    isPc = isPc.astype(int)
    LEFsubset = LEFpositions[t*restartSimulationEveryBlocks:(t+1)*restartSimulationEveryBlocks,:,:]
    milker = bondUpdater(LEFsubset)
    a = Simulation(N=N, 
                   error_tol=0.01, 
                   collision_rate=0.02, 
                   integrator ="variableLangevin", 
                   platform="cuda",
                   GPU = "0", 
                   PBCbox=(PBC_width, PBC_width, PBC_width),
                   reporters=[reporter],
                   precision="mixed")  # platform="CPU", 
    a.set_data(data) # initial polymer 
    a.add_force(
        polychrom.forcekits.polymer_chains(
            a,
            chains=chains,
            nonbonded_force_func=polychrom.forces.heteropolymer_SSW,
            nonbonded_force_kwargs={
                'attractionEnergy': 0,  # base attraction energy for all monomers
                'attractionRadius': attraction_radius,
                'interactionMatrix': interactionMatrix,
                'monomerTypes': isPc,  # the updated colors 
                'extraHardParticlesIdxs': []
            },
            bond_force_kwargs={
                'bondLength': 1,
                'bondWiggleDistance': 0.05
            },
            angle_force_kwargs={
                'k': 1.5
            }
        )
    )
    # ------------ initializing milker; adding bonds ---------
    # copied from addBond
    kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
    bondDist = smcBondDist * a.length_scale
    activeParams = {"length":bondDist,"k":kbond}
    inactiveParams = {"length":bondDist, "k":0}
    milker.setParams(activeParams, inactiveParams)  
    # this step actually puts all bonds in and sets first bonds to be what they should be
    milker.setup(bondForce=a.force_dict['harmonic_bonds'],
                blocks=restartSimulationEveryBlocks)

    # Start simulation without local energy minimization 
    a._apply_forces()

    for i in range(restartSimulationEveryBlocks):        
        if i % saveEveryBlocks == (saveEveryBlocks - 1):  
            a.do_block(steps=steps)
        else:
            a.integrator.step(steps)  # do steps without getting the positions from the GPU (faster)
        if i < restartSimulationEveryBlocks - 1: 
            curBonds, pastBonds = milker.step(a.context)  # this updates bonds. You can do something with bonds here
    data = a.get_data()  # save data and step, and delete the simulation
    del a    
    reporter.blocks_only = True  # Write output hdf5-files only for blocks
    time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)
    reporter.dump_data()
    
    saveColors = saveFolder + 'colorStates.csv'
    pd.DataFrame(colorStates).to_csv(saveColors)
