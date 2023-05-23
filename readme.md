RRC Conductance Profile Manuscript Supplimentary Material

supercell_ga.py - Python script containing the classes and functions for the developed genetic algorithm.

run_ga.py - A python script to run supercell_ga.py. The current set up runs the GA with 5 individuals and 3 generations as a small example. 

get_ap_data.py - A python script which runs simulations in the ToR-ORd and Grandi models to generate action potential data. Specically, this file generates the data in the following manuscript figures: 4A, 5, 6B, 7, 8A, and 8B. The data is stored as a pickle file contained in the data/ folder and named cond_data.pkl.

rrc_exp.py - A python script which runs simulations to calculate the change in action potential duraction between the baseline ToR-ORd model and the 220 best GA individuals at various stimuli. This data was used to produce Figures 3D and 3E. The data is stored as a csv file contained in the data/ folder and named rrc_data.csv.

important_functions.py - A python script containing functions used in algorithm development, experiments, and analysis. Since these are commonly used, they are kept in this one file and loaded into additional files when needed. 

reproduce_figures.ipynb - Jupyter notebook file containing all code necissary to reproduce the manuscript's figures.

figures/ - a folder containing all the main and supplimentary figures included in the manuscript. Each figure is saved as both .png and .pdf files. 

data/ - A folder where all required data for analysis and figure production is kept. This folder contains 7 data files:
    - all_data.csv.gz is a compressed dataframe which contains conductance, cost, and RRC data for every individual from all eight GA runs. The GA trial and generation is marked for each and the ToR-ORd model was used for all simulations. 
    - APbounds.csv is the upper and lower bound data used to assess action potential morphology point by point after upstroke. 
    - baseline_torord_data.csv is a dataframe containing time and voltage data to produce the baseline ToR-ORd model used in this study. 
    - best_data.csv is a dataframe containing the same information as all_data.csv.gz but only for the 220 best individuals. These all satisfy physiologic constraints and have a cost lower than 2800. 
    - cond_data.pkl - a pickle file produced from the get_ap_data.py script. 
    - fig2_data.csv.gz - a compressed dataframe which contains conductance, cost, RRC, feature, action potential, and calcium transient data for all individuals from generations 0, 30, and 99 of all trials. This data is used to produce Figure 2B and 2C. 
    - grandi_pop.csv.gz -  a compressed dataframe which contains conductance, cost, RRC, action potential, calcium transient, and feature data for a population of 200 Grandi models. This data was used for a multiparameter sensitivity analysis (Figure 8C). 
    - rrc_data.csv - a dataframe produced from the rrc_exp.py file. This file contains the conductance information for each of the 220 best individuals as well the time, voltage, and deltaAPD90 values at 5 stimuli: 0A/F, 0.05A/F, 0.1A/F, 0.15A/F, and 0.2A/F. 

models/ - a folder containing all models used in algorithm training and analysis. This folder contains 3 files:
    - grandi_flat.mmt is the endocardial version of the Grandi et al model. The only change made was the inclusion of the Ibias current. This model was used to produce the AP and current traces in Figure 7A of the manuscript in addition to GBM1, GBM2, and GBM3 of Figure 8A and 8B. 
    - grandi_flat_NaL.mmt is the same as the grandi_flat.mmt model with the ToR-ORd INaL formulation included. This model was used to produce GBM4 in Figure 8A and 8B of the manuscript. 
    - tor_ord_endo2.mmt is the endocardial version of the Tomek et al model. The only change made was the inclusion of the Ibias current. This model was used for all simulations in GA training and much of the analysis. 



# RRC-ConductanceProfile-Project
# RRC-ConductanceProfile-Project
