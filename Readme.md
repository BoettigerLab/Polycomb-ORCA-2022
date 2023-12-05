This repository contains code to run the simulations described in our work to understand possible mechanisms underlying the distinctive organization of polycomb-bound chromatin at hox gene-complexes.

[![DOI](https://zenodo.org/badge/555168393.svg)](https://zenodo.org/doi/10.5281/zenodo.10258160)

Polycomb repression of Hox genes involves spatial feedback but not domain compaction or demixing

Sedona Eve Murphy<sup>1,2</sup> and Alistair Nicol Boettiger<sup>1</sup>

<sup>1</sup>Department of Developmental Biology, Stanford University
<sup>2</sup>Department of Genetics, Stanford University 

Correspondence: boettiger@stanford.edu 

The intention of this repository is to make the transparent the workflow behind the data analysis and behind the simulations.  Included here:

An ipython notebook detailing the steps used in the polymer simulations of polycomb chromatin spreading and 3D motion of polycomb bound chromatin.  

To run the polymer simulations, this code requires the open-access [polychrom repository](https://github.com/open2c/polychrom) developed by the open2c team.  This code is tested to run with our local open-access fork of the project, [https://github.com/BoettigerLab/polychrom](https://github.com/BoettigerLab/polychrom).  This fork contains a few added functions to explore distinct behaviors like targeted loading of cohesin and optional tethering of polymer ends to a spherical confining lamina, which were not included in the original repository at the time of our fork. For compatibility we recommend running simulations with our fork, but it should work with either as the added functionalities are not exploited in this work on polycomb chromatin. 

The chromatin tracing image data is deposited at the [4DN data portal](https://data.4dnucleome.org/) and can be retrieved there, following publication and public release.  We will update the link here when it is available.  

