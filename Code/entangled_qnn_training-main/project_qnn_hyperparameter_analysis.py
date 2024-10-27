import csv
import gc
import json
import time
from datetime import datetime
from qnns.cuda_qnn import CudaPennylane
import re

from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from victor_thesis_experiments_main import *
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from sgd_for_scipy import *
from jax import jacrev
import os
import pandas as pd
from scipy.optimize import minimize, dual_annealing
import re

from project_qnn_analysis import *

databatches = ["databatch_0", "databatch_1", "databatch_2", "databatch_3", "databatch_4"]


def load_fun_nit_per_hyperparameter_data(data, opt, hyperparameter):
    '''
        For a specific optimizer {opt}: Creates one dictionary that contains a list of achieved function values after optimization
        per value for a hyperparameter and another dictionary with numer of iterations needed.
        
        Example: For Genetic Algorithm, hyperparameter "parent_selection_type" has values ["sss", "rws", "tournament", "rank"]
    '''
    fun_per_hyperparameter_value = {}
    nit_per_hyperparameter_value = {}

    #choose correct dictionary key names in json file for a specific optimizer
    fun_key_name = "fun"
    nit_key_name = "nit"

    # genetic algorithm uses "generation" instead of "iteration"
    if(opt=="genetic_algorithm"):
        nit_key_name = "ngeneration/max_iter"
    n=0


    for i in range(len(data)):
        for databatch_id in databatches:
            try:
                dict = data[i][databatch_id][opt]
                for j in range(0,len(dict)-1):
                    hyperparameter_value = dict[str(j)][hyperparameter]

                    if hyperparameter_value not in fun_per_hyperparameter_value:
                        fun_per_hyperparameter_value[hyperparameter_value] = []
                    if hyperparameter_value not in nit_per_hyperparameter_value:
                        nit_per_hyperparameter_value[hyperparameter_value] = []

                    #append fun and nit value to correct list in result dictionaries
                    fun_per_hyperparameter_value[hyperparameter_value].append(float(dict[str(j)][fun_key_name]))
                    nit_per_hyperparameter_value[hyperparameter_value].append(int(dict[str(j)][nit_key_name]))
            except KeyError as e:
                print(f"Fehler beim Lesen der Daten: {e}")
    return fun_per_hyperparameter_value, nit_per_hyperparameter_value

def load_fun_nit_for_c1_c2_PSO(data):
    '''
        c1 and c2 values are saved separately in the generated json file, however they need to be analysed together.
        
        Creates one dictionary that contains a list of achieved function values after optimization
        per value for a (c1,c2) pair and another dictionary with number of iterations needed.
    '''
    fun_per_hyperparameter_value = {}
    nit_per_hyperparameter_value = {}

    #choose correct dictionary key names in json file for a specific optimizer
    fun_key_name = "fun"
    nit_key_name = "nit"

    for i in range(len(data)):
        for databatch_id in databatches:
            try:
                dict = data[i][databatch_id][opt]
                for j in range(0,len(dict)-1):
                    c1 = dict[str(j)]["c1"]
                    c2 = dict[str(j)]["c2"]
                    c1_c2_value = f"[{c1},{c2}]"
                    if c1_c2_value not in fun_per_hyperparameter_value:
                        fun_per_hyperparameter_value[c1_c2_value] = []
                    if c1_c2_value not in nit_per_hyperparameter_value:
                        nit_per_hyperparameter_value[c1_c2_value] = []

                    #append fun and nit value to correct list in result dictionaries
                    fun_per_hyperparameter_value[c1_c2_value].append(dict[str(j)][fun_key_name]) 
                    nit_per_hyperparameter_value[c1_c2_value].append(dict[str(j)][nit_key_name])
            except KeyError as e:
                print(f"Fehler beim Lesen der Daten: {e}")
    return fun_per_hyperparameter_value, nit_per_hyperparameter_value


def create_hyperparameter_boxplots(path,json_data, opt, hyperparameters):
    # create path if it does not exist 
    os.makedirs(path, exist_ok=True)
    # replace "iterations" with "generations" in plots if opt is genetic algorithm
    nit_name = "iterations"
    if opt == "genetic algorithm":
        nit_name = "generations"
    # make two boxplots per hyperparameter: one for function values, one for number of iterations
    for par in hyperparameters:
        par_name = par
        if par == "max_generation/_iter":
            par_name = "max_iter" 

        #c1,c2 need to be analysed separately
        if par == "c1_c2":
            fun_dict, nit_dict = load_fun_nit_for_c1_c2_PSO(json_data)
        else:
            fun_dict, nit_dict = load_fun_nit_per_hyperparameter_data(json_data,opt,par)

        # Boxplot for function values
        file_path = os.path.join(path, f'{opt}_boxplot_fun_{par_name}.png')
        plt.figure()
        plt.boxplot(fun_dict.values())
        plt.xticks(range(1, len(fun_dict.keys()) + 1), fun_dict.keys())
        plt.xlabel(par_name)
        plt.ylabel('Function value')
        plt.title(f"Achieved loss function values per values of \n {par_name} for {opt}")
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        # Boxplot for number of iterations
        file_path = os.path.join(path, f'{opt}_boxplot_nit_{par_name}.png')
        plt.figure()
        plt.boxplot(nit_dict.values())
        plt.xticks(range(1, len(nit_dict.keys()) + 1), nit_dict.keys())
        plt.xlabel(par_name)
        plt.ylabel(f'Number of {nit_name}')
        plt.title(f"Number of {nit_name} per values of \n {par_name} for {opt}")
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

if __name__ == "__main__":
    directory = "experimental_results/results/optimizer_results/hyperparameter_tests"
    opt_list = ["genetic_algorithm", "particle_swarm"]
    json_data = load_json_files(directory)
    hyperparameters_per_opt = {"genetic_algorithm": ["max_generation/_iter","parent_selection_type", "crossover_type", "mutation_type"], 
                                "particle_swarm": ["maxiter", "n_particles", "w", "ftol", "c1_c2"],
                                "diff_evolution": ["maxiter", "recombination", "popsize", "tol"]}

    # create top and bottom 25 hyperparameter combinations for GA and PSO

    # create boxplots for DE
    opt = "diff_evolution"
    save_path = f'qnn-experiments/experimental_results/results/hyperparameter_boxplots/{opt}'
    create_hyperparameter_boxplots(save_path,json_data,opt,hyperparameters_per_opt[opt])

    # DONE: create boxplots for GA and PSO
    # for opt in opt_list:
    #     save_path = f'qnn-experiments/experimental_results/results/hyperparameter_boxplots/{opt}'
    #     create_hyperparameter_boxplots(save_path,json_data,opt,hyperparameters_per_opt[opt])

    
    