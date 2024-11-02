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
conf_ids_to_skip = [190, 191, 192, 193, 194, 210, 211, 212, 213, 214, 230, 231, 232, 233, 234]
hyperparameters_per_opt = {"genetic_algorithm": ["maxiter", "crossover_type", "stop_criteria"], 
                                "particle_swarm": ["maxiter", "ftol"],
                                "diff_evolution": ["maxiter"],
                                "nelder_mead": ["maxiter", "fatol", "xatol"],
                                "bfgs": ["maxiter", "gtol", "xrtol", "eps"],
                                "cobyla": ["maxiter", "tol","catol"],
                                "powell": ["maxiter", "ftol", "xtol"],
                                "slsqp": ["maxiter", "ftol", "eps"],
                                "sgd": ["maxiter","learning_rate","eps"],
                                "rmsprop": ["maxiter", "learning_rate", "eps"],
                                "adam": ["maxiter", "learning_rate", "eps"],
                                "dual_annealing": ["maxiter"]
                                }


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

    if opt == "cobyla":
        nit_key_name = "nfev"

    for i in range(len(data)):
        conf_id = data[i]["conf_id"]
        if conf_id in conf_ids_to_skip:
            continue
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
                    if(float(dict[str(j)][fun_key_name]) < 0):
                        print("config ",conf_id, "databatch ", databatch_id, "run_n", j,hyperparameter, hyperparameter_value, "FUN WERT: ", dict[str(j)][fun_key_name])
            except KeyError as e:
                print(f"Fehler beim Lesen der Daten: {e}")
    return fun_per_hyperparameter_value, nit_per_hyperparameter_value

def load_fun_nit_for_c1_c2_PSO(data):
    '''
        c1 and c2 values are saved separately in the generated json file, however they need to be analysed together.
        
        Creates one dictionary that contains a list of achieved function values after optimization
        per value for a (c1,c2) pair and another dictionary with number of iterations needed.
    '''
    opt = "particle_swarm"
    fun_per_hyperparameter_value = {}
    nit_per_hyperparameter_value = {}

    #choose correct dictionary key names in json file for a specific optimizer
    fun_key_name = "fun"
    nit_key_name = "nit"

    for i in range(len(data)):
        conf_id = data[i]["conf_id"]
        if conf_id in conf_ids_to_skip:
            continue
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
    nit_name_short = "nit"
    if(opt == "cobyla"):
        nit_name = "objective function evaluations"
        nit_name_short = "nfev"
    # make two boxplots per hyperparameter: one for function values, one for number of iterations
    for par in hyperparameters:
        #c1,c2 need to be analysed separately
        if par == "c1_c2":
            fun_dict, nit_dict = load_fun_nit_for_c1_c2_PSO(json_data)
        else:
            fun_dict, nit_dict = load_fun_nit_per_hyperparameter_data(json_data,opt,par)

        # Boxplot for function values
        file_path = os.path.join(path, f'{opt}_boxplot_fun_{par}.png')
        plt.figure()
        plt.boxplot(fun_dict.values())
        plt.xticks(range(1, len(fun_dict.keys()) + 1), fun_dict.keys())
        plt.xlabel(par)
        plt.ylabel('Function value')
        plt.title(f"Achieved loss function values per values of \n {par} for {opt}")
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        # Boxplot for number of iterations
        file_path = os.path.join(path, f'{opt}_boxplot_{nit_name_short}_{par}.png')
        plt.figure()
        plt.boxplot(nit_dict.values())
        plt.xticks(range(1, len(nit_dict.keys()) + 1), nit_dict.keys())
        plt.xlabel(par)
        plt.ylabel(f'Number of {nit_name}')
        plt.title(f"Number of {nit_name} per values of \n {par} for {opt}")
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

def get_all_fun_values_for_opts(data, opt_list):
    optimizer_data = {}
    for i in range(len(data)):
        conf_id = data[i]["conf_id"]
        if conf_id in conf_ids_to_skip:
            continue
        for databatch_id in databatches:
            try:
                for opt in opt_list:
                    if opt not in optimizer_data:
                        optimizer_data[opt] = []
                    dict = data[i][databatch_id][opt]
                    for j in range(0,len(dict)-1):
                        #append fun value to correct list in result dictionaries
                        optimizer_data[opt].append(float(dict[str(j)]["fun"]))
            except KeyError as e:
                print(f"Fehler beim Lesen der Daten: {e}")
    return optimizer_data

def all_opts_fun_value_boxplots(path,json_data, opt_list):
    '''
        TODO: anpassen, dass man Ergebnisse von allen Experimenten (i.e. alle Optimierer) hat. 
        TODO: Speicherplatz und Zeit(?) sparen indem man nicht alle json files in einem dictionary speichert
    '''
    fun_dict = get_all_fun_values_for_opts(json_data,opt_list)
    # Boxplot for function values
    file_path = os.path.join(path, f'all_opt_boxplot_fun.png')
    plt.figure()
    plt.boxplot(fun_dict.values())
    plt.xticks(range(1, len(fun_dict.keys()) + 1), fun_dict.keys())
    plt.xlabel("Optimizer")
    plt.ylabel('Function value')
    plt.title(f"Achieved loss function values for different Optimizers")
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

def create_all_hyperparameter_boxplots():
    '''
        Creates boxplot for each optimizer based on Data from final Experiment run.
        For each hyperparameter for each optimizer two boxplots are created: 
            one for the distribution of the achieved function value for each value of this hyperparameter and
            one for the distribution of the needed iterations for each value of this hyperparameter.
        
        Beware: directories for experiment result json-files must be correct.
        Beware: At least 7GB RAM are needed to run this.
    '''
    # experiment part 1: nelder_mead, bfgs, cobyla, powell, slsqp, sgd, rmsprop, adam
    directory = "experimental_results/results/optimizer_results/experiment_part1"
    opt_list = ["nelder_mead","bfgs","cobyla","powell","slsqp","sgd","rmsprop","adam","dual_annealing"]
    json_data = load_json_files(directory)
    # create boxplots for experiment part 1
    for opt in opt_list:
        save_path = f'qnn-experiments/plots/box_plots/hyperparameter_boxplots/{opt}'
        create_hyperparameter_boxplots(save_path,json_data,opt,hyperparameters_per_opt[opt])
        print(f"{opt} done")
    #all_opts_fun_value_boxplots(save_path,json_data,opt_list)
    del json_data
    # experiment part 1: nelder_mead, bfgs, cobyla, powell, slsqp, sgd, rmsprop, adam
    directory = "experimental_results/results/optimizer_results/experiment_part2_GA_PSO_DE"
    opt_list = ["genetic_algorithm", "particle_swarm", "diff_evolution"]
    json_data = load_json_files(directory)
    # create boxplots for experiment part 1
    for opt in opt_list:
        save_path = f'qnn-experiments/plots/box_plots/hyperparameter_boxplots/{opt}'
        create_hyperparameter_boxplots(save_path,json_data,opt,hyperparameters_per_opt[opt])
        print(f"{opt} done")
    del json_data

if __name__ == "__main__":
    start = time.time()
    print(f"start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")

    create_all_hyperparameter_boxplots()

    end = time.time()
    print(f"end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
    print(f"total runtime (with callback): {np.round((end-start)/60,2)}min") 
    



    
    