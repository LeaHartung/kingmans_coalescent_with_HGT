# kingmans_coalescent_with_HGT

## Description

This repository provides code to simulate a Kingman's coalescent with horizontal gen transfer (HGT). It provides 
code to simulate two types of HGT, uniform and distance dependent. Additionally, it includes scripts to compare the 
resulting trees among each other.

## Requirements

The code was created in Python 3.10 on a Windows machine.
The required modules are listed in requirements.txt.

## Usage
The functions to simulate the species and gene trees as well as the functions to evaluate the tree distances are 
provided in the `simulation/` directory. The trees created by the simulations are stored in Newick format (with 
distances and leaf names). To calculte their distances we use the packages `ete3`, `PhyoDM`, and `DendroPy`.  

Scripts to run the simulation and evaluation are provided in the directory `scripts/`. There we provided the 
following scripts
 - `run_simulations.py` to systematically run the simulation for a range of parameters and a set number of repetitions
 - `evaluate_distances.py` to calculate and safe the distances between the species and gene trees
 - `plotting.py` to create plots from the results of `evaluate_distances.py` 
 - `example.py` to provide an example for a single species process with the corresponding distance evaluations

The workflow to run extensive simulations is as follows:

First define the simulation variables such as speciation rate, HGT rate, number of species, and sample size , in 
the `main` of `run_simulations.py`. Running the script will create a subdirectory in `results/`, in which the 
simulated trees, their times to the most recent common ancestor (MRCA) as well as the simulation parameters are saved.

Then run `evaluate_distances.py`, providing the name of the subdirectory in `results/`. This will create files with 
the average distances between the species and gene trees as well as between gene trees in the same species tree (if 
more than one was simulated) in the given subdirectory. 

Finally, running `plotting.py` will create plots regarding the time to the MRCA and the calculated tree distances 
for a given subdirectory.