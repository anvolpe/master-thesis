import time
import os
from datetime import datetime
from qnns.cuda_qnn import CudaPennylane

from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from victor_thesis_experiments_main import *
from concurrent.futures import ProcessPoolExecutor
from project_qnn_sgd_for_scipy import *
from project_qnn_analysis import *

'''
    Analysis of hyperparameters for all optimizers (using boxplots)
'''

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

hyperparameters_per_opt_prelim = {"genetic_algorithm": ["crossover_type", "parent_selection_type", "mutation_type"], 
                                "particle_swarm": ["maxiter", "ftol", "n_particles", "c1_c2", "w"],
                                "diff_evolution": ["maxiter", "popsize", "recombination"]
                                }

opt_titles = {'nelder_mead': 'Nelder-Mead', 'powell':'Powell', 'sgd':'SGD', 
              'adam':'Adam', 'rmsprop':'RMSprop', 'bfgs':'BFGS','slsqp':'SLSQP',
              'dual_annealing':'Dual Annealing','cobyla':'COBYLA',
              'genetic_algorithm':'Genetic Algorithm', 'particle_swarm': 'Particle Swarm Optimization',
              'diff_evolution':'Differential Evolution'}


def load_fun_nit_per_bounds_data(opt, prelim=False):
    '''
        For a specific optimizer {opt}: Creates one dictionary that contains a list of achieved function values after optimization
        per interval for the hyperparameter bounds and another dictionary with number of iterations needed.
        Data source: "experimental_results/results/optimizer_results/bounds_2024-07-29"

        Arguments:
            opt (String): optimizer name
            prelim (boolean, optional): true if underlying data is from preliminary tests

        Returns:
            fun_per_bounds (dict): keys are a string of form "bounds_[n] (n between 0 and 4), values a list of achieved function values for this optimizer and bounds
            nit_per_bounds (dict): keys are a string of form "bounds_[n] (n between 0 and 4), values a list of number of iterations for this optimizer and bounds
    '''
    fun_per_bounds = {}
    nit_per_bounds = {}
    #load json data
    directory = "experimental_results/results/optimizer_results/bounds_2024-07-29"
    data = load_json_data(directory)
    bounds_values = {"bounds_0": "none", "bounds_1": "$[0,2\pi]$", "bounds_2": "$[0,4\pi]$", "bounds_3": "$[-2\pi, 2\pi]$", "bounds_4": "$[-4\pi, 4\pi]$"}

    #choose correct dictionary key names in json file for a specific optimizer
    fun_key_name = "fun"
    nit_key_name = "nit"


    if opt == "cobyla":
        nit_key_name = "nfev"

    for i in range(len(data)):
        conf_id = data[i][0]["conf_id"]
        if conf_id in conf_ids_to_skip:
            continue
        for j in range(len(data[i])):
            d = data[i][j]
            for databatch_id in databatches:
                for bounds_id in bounds_values.keys():
                    try:
                        dict = d[databatch_id][bounds_id][opt]
                        bounds_value = bounds_values[bounds_id]
                        if bounds_value not in fun_per_bounds:
                            fun_per_bounds[bounds_value] = []
                        if bounds_value not in nit_per_bounds:
                            nit_per_bounds[bounds_value] = []

                        for j in range(0,len(dict)-1):
                            #append fun and nit value to correct list in result dictionaries
                            fun_per_bounds[bounds_value].append(float(dict[str(j)][fun_key_name]))
                            nit_per_bounds[bounds_value].append(int(dict[str(j)][nit_key_name]))
                    except KeyError as e:
                        print(f"Fehler beim Lesen der Daten: {bounds_id}: {e}")
    return fun_per_bounds, nit_per_bounds

def load_fun_nit_per_hyperparameter_data(data, opt, hyperparameter, prelim=False,maxiter=False):
    '''
        For a specific optimizer {opt}: Creates one dictionary that contains a list of achieved function values after optimization
        per value for a hyperparameter and another dictionary with numer of iterations needed.
        
        Example: For Genetic Algorithm, hyperparameter "parent_selection_type" has values ["sss", "rws", "tournament", "rank"]

        Arguments:
            data (dict): contains all data from several json files
            opt (String): optimizer name
            hyperparameter (String): 
            prelim (boolean, optional): true if underlying data is from preliminary tests
            maxiter (boolean, optional): true if only nit values for maxiter=1000 should be saved

        Returns:
            fun_per_hyperparameter_value (dict): keys are a string of form "bounds_[n] (n between 0 and 4), values a list of achieved function values for this optimizer and hyperparameter values
            nit_per_hyperparameter_value (dict): keys are a string of form "bounds_[n] (n between 0 and 4), values a list of number of iterations for this optimizer and hyperparameter values
    '''
    fun_per_hyperparameter_value = {}
    nit_per_hyperparameter_value = {}

    #choose correct dictionary key names in json file for a specific optimizer
    fun_key_name = "fun"
    nit_key_name = "nit"
    maxiter_key_name = "maxiter"
    if prelim==True and opt == "genetic_algorithm":
        nit_key_name = "ngeneration/max_iter"
        maxiter_key_name = "max_generation/_iter"
        
    if opt == "cobyla":
        nit_key_name = "nfev"
    
    i_range = np.arange(len(data))
    if prelim==True:
        i_range = i_range*5 # if preliminary tests: every 5th config

    for i in i_range:
        conf_id = data[i][0]["conf_id"]
        if conf_id in conf_ids_to_skip:
            continue
        for j in range(len(data[i])):
            d = data[i][j]
            for databatch_id in databatches:
                try:
                    dict = d[databatch_id][opt]
                    for j in range(0,len(dict)-2):
                        hyperparameter_value = dict[str(j)][hyperparameter]
                        maxiter_value = dict[str(j)][maxiter_key_name]
                        if hyperparameter_value not in fun_per_hyperparameter_value:
                            fun_per_hyperparameter_value[hyperparameter_value] = []
                        if hyperparameter_value not in nit_per_hyperparameter_value:
                            nit_per_hyperparameter_value[hyperparameter_value] = []

                        #append fun and nit value to correct list in result dictionaries
                        fun_per_hyperparameter_value[hyperparameter_value].append(float(dict[str(j)][fun_key_name]))
                        if(maxiter==True): # only add nit value for maxiter=1000
                            if(maxiter_value==1000):
                                nit_per_hyperparameter_value[hyperparameter_value].append(int(dict[str(j)][nit_key_name]))
                        else:
                            nit_per_hyperparameter_value[hyperparameter_value].append(int(dict[str(j)][nit_key_name]))
                        if(float(dict[str(j)][fun_key_name]) < 0):
                            print("config ",conf_id, "databatch ", databatch_id, "run_n", j,hyperparameter, hyperparameter_value, "FUN WERT: ", dict[str(j)][fun_key_name])
                except KeyError as e:
                    print(f"Fehler beim Lesen der Daten: {e}")
    return fun_per_hyperparameter_value, nit_per_hyperparameter_value

def load_fun_per_n_particles_maxiter_pso(data, prelim=False):
    '''
        For Particle Swarm Optimization: Creates one dictionary. Restriction: only ftol=-inf, i.e. no premature termination.
        Level 1 Key: number of particles, Level 2 Key: maxiter, Values: list of achieved function values for that combination.

        Arguments:
            data (dict): contains all data from several json files
            prelim (boolean, optional): true if underlying data is from preliminary tests
        
        Returns: 
            fun_per_hyperparameter_value (dict): Level 1 Key: number of particles, Level 2 Key: maxiter, Values: list of achieved function values for that combination.

    '''
    fun_per_hyperparameter_value = {}

    #choose correct dictionary key names in json file for a specific optimizer
    fun_key_name = "fun"
    nit_key_name = "nit"
    opt = "particle_swarm"
    conf_list = np.arange(0,320,1)
    if(prelim==True):
        conf_list = np.arange(0,320,5)
    for i in conf_list:
        conf_id = data[i][0]["conf_id"]
        if conf_id in conf_ids_to_skip:
            continue
        for j in range(len(data[i])):
            d = data[i][j]
            for databatch_id in databatches:
                try:
                    dict = d[databatch_id][opt]
                    for j in range(0,len(dict)-2):
                        n_particles = dict[str(j)]["n_particles"]
                        ftol = dict[str(j)]["ftol"]
                        maxiter = dict[str(j)]["maxiter"]
                        if n_particles not in fun_per_hyperparameter_value:
                            fun_per_hyperparameter_value[n_particles] = {}
                        if maxiter not in fun_per_hyperparameter_value[n_particles]:
                            fun_per_hyperparameter_value[n_particles][maxiter] = []

                        #append fun and nit value to correct list in result dictionaries
                        fun_per_hyperparameter_value[n_particles][maxiter].append(float(dict[str(j)][fun_key_name]))
                except KeyError as e:
                    print(f"Fehler beim Lesen der Daten: {e}")
    return fun_per_hyperparameter_value


def create_multi_PSO_boxplot(save_path):
    '''
        Creates one boxplot where x-axis is maxiter value and colored boxplots are number of particles.

        Arguments:
            save_path (Sring): save path for plot
    '''
    directory = "experimental_results/results/optimizer_results/hyperparameter_tests_2024-10-26"
    id_list = np.arange(0,320,5)
    data = load_json_data(directory, conf_id_list=id_list)
    title =  "Particle Swarm Optimization:\n Achieved loss function values per maxiter and number of particles"
    fun_values = load_fun_per_n_particles_maxiter_pso(data,prelim=True)
    plot_boxplots(save_path,title,fun_values[10],fun_values[30],fun_values[60],"maxiter",[100, 500, 1000],labels=["10 particles", "30 particles", "60 particles"])

def load_fun_nit_for_c1_c2_PSO(data, prelim=False):
    '''
        c1 and c2 values are saved separately in the generated json file, however they need to be analysed together.
        
        Creates one dictionary that contains a list of achieved function values after optimization
        per value for a (c1,c2) pair and another dictionary with number of iterations needed.

        Arguments:
            data (dict): contains all data from several json files
            prelim (boolean, optional): true if underlying data is from preliminary tests   

        Returns:
            fun_per_hyperparameter_value (dict): keys are a string of form "bounds_[n] (n between 0 and 4), values a list of achieved function values for this optimizer and c1,c2 values
            nit_per_hyperparameter_value (dict): keys are a string of form "bounds_[n] (n between 0 and 4), values a list of number of iterations for this optimizer and c1,c2 values 

    '''
    opt = "particle_swarm"
    fun_per_hyperparameter_value = {}
    nit_per_hyperparameter_value = {}

    #choose correct dictionary key names in json file for a specific optimizer
    fun_key_name = "fun"
    nit_key_name = "nit"

    if prelim==True and opt == "genetic_algorithm":
        nit_key_name = "ngeneration/max_iter"

    i_range = np.arange(len(data))
    if prelim==True:
        i_range = i_range*5 # if preliminary tests: every 5th config

    for i in i_range:
        conf_id = data[i][0]["conf_id"]
        if conf_id in conf_ids_to_skip:
            continue
        for j in range(len(data[i])):
            d = data[i][j]
            for databatch_id in databatches:
                try:
                    dict = d[databatch_id][opt]
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


def create_hyperparameter_boxplots(path,json_data, opt, hyperparameters, prelim=False, more_info=False,maxiter=False):
    '''
        Creates boxplots for each hyperparameter in hyperparameters for optimizer opt. Json Data source is json_data. 
        If prelim=True: keys in json files are a bit different for preliminary testing for some optimizers
        If more_info=True: More information, such as mean and STD are save in a txt-file called 
            {opt}_hyperparameter_info.txt in the same location

        Arguments:
            path (String): save path for boxplots
            json_data (dict): contains all data from several json files
            opt (String): name of optimizer
            hyperparameters (list of Strings): list of hyperparameters for optimizer opt
            prelim (boolean, optional): true if underlying data is from preliminary tests  
            more_info (boolean, optional): true if further information on boxplots (such as median, min, max, std) should be saved in txt file

    '''
    # create path if it does not exist 
    os.makedirs(path, exist_ok=True)
    # prep txt file
    file_text = f"{opt} hyperparameter info\n===============================\n"
    # replace "iterations" with "generations" in plots if opt is genetic algorithm
    nit_name = "iterations"
    nit_name_short = "nit"
    if(opt == "cobyla"):
        nit_name = "objective function evaluations"
        nit_name_short = "nfev"
    # make two boxplots per hyperparameter: one for function values, one for number of iterations
    text = {}
    for par in hyperparameters:
        #c1,c2 need to be analysed separately
        if par == "c1_c2":
            fun_dict, nit_dict = load_fun_nit_for_c1_c2_PSO(json_data, prelim=prelim)
        elif par == "bounds":
            fun_dict, nit_dict = load_fun_nit_per_bounds_data(opt, prelim=prelim)
        else:
            fun_dict, nit_dict = load_fun_nit_per_hyperparameter_data(json_data,opt,par, prelim=prelim,maxiter=maxiter)
        if(more_info==True):
            text[par] = ""
            for value in fun_dict.keys():
                median = np.median(fun_dict[value])
                min = np.min(fun_dict[value])
                max = np.max(fun_dict[value])
                std = np.std(fun_dict[value])
                text[par] += f"{par} = {value}: median={median},    min={min},  max={max},  std={std}\n"
        # Boxplot for function values
        file_path = os.path.join(path, f'{opt}_boxplot_fun_{par}.png')
        plt.figure(figsize=(12.8,9.6))
        plt.boxplot(fun_dict.values())
        plt.xticks(range(1, len(fun_dict.keys()) + 1), fun_dict.keys(),fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlabel(par,fontsize=28)
        plt.ylabel('Function value',fontsize=28)
        plt.title(f"Achieved loss function values per values of \n {par} for {opt_titles[opt]}",fontsize=30)
        plt.grid(True)
        plt.savefig(file_path, dpi=1200)
        plt.close()


        # Boxplot for number of iterations
        file_path = os.path.join(path, f'{opt}_boxplot_{nit_name_short}_{par}.png')
        plt.figure(figsize=(12.8,9.6))
        plt.boxplot(nit_dict.values())
        plt.xticks(range(1, len(nit_dict.keys()) + 1), nit_dict.keys(),fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlabel(par,fontsize=28)
        plt.ylabel(f'Number of {nit_name}',fontsize=28)
        title = f"Number of {nit_name} per values of \n {par} for {opt_titles[opt]}"
        if(maxiter==True):
            title += ", maxiter=1000"
        plt.title(title,fontsize=28)
        plt.grid(True)
        plt.savefig(file_path,dpi=1200)
        plt.close()

    if(more_info==True):
        for par in hyperparameters:
            file_text += text[par]+"===============================\n"
        save_path = os.path.join(path, f'{opt}_hyperparameter_info.txt')
        with open(save_path, 'w') as f:
            f.write(file_text)

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
    json_data = load_json_data(directory)
    # create boxplots for experiment part 1
    for opt in opt_list:
        save_path = f'qnn-experiments/plots/hyperparameter_plots/final_experiment/{opt}'
        create_hyperparameter_boxplots(save_path,json_data,opt,hyperparameters_per_opt[opt],more_info=True)
        print(f"{opt} done")
    #all_opts_fun_value_boxplots(save_path,json_data,opt_list)
    del json_data
    # experiment part 2: Genetic Algorihtm, PSO, Differential Evolution
    directory = "experimental_results/results/optimizer_results/experiment_part2_GA_PSO_DE"
    opt_list = ["genetic_algorithm", "particle_swarm", "diff_evolution"]
    json_data = load_json_data(directory)
    # create boxplots for experiment part 2
    for opt in opt_list:
        save_path = f'qnn-experiments/plots/hyperparameter_plots/final_experiment/{opt}'
        create_hyperparameter_boxplots(save_path,json_data,opt,hyperparameters_per_opt[opt],more_info=True)
        print(f"{opt} done")
    del json_data

def create_preliminary_test_boxplots():
    '''
        Create hyperparameter boxplots for preliminary tests for Dual Annealing, PSO, Genetic Algorithm and Differential Evolution
    '''
    directory = "experimental_results/results/optimizer_results/bounds_2024-07-29"
    save_path = "qnn-experiments/plots/hyperparameter_plots/preliminary_test/bounds/dual_annealing"
    json_data = load_json_data(directory)
    create_hyperparameter_boxplots(save_path,json_data, "dual_annealing", ["bounds"], prelim=False, more_info=False)
    del json_data

    directory = "experimental_results/results/optimizer_results/hyperparameter_tests_2024-10-26"
    json_data = load_json_data(directory, conf_id_list=range(0,320,5))
    print(len(json_data))
    opt_list = ["genetic_algorithm","particle_swarm","diff_evolution"]
    for opt in opt_list:
        save_path = f'qnn-experiments/plots/hyperparameter_plots/preliminary_test/hyperparameters_GA_DE_PSO/{opt}'
        create_hyperparameter_boxplots(save_path,json_data,opt,hyperparameters_per_opt_prelim[opt],prelim=True,more_info=True,maxiter=True)
        print(f"{opt} done")
    del json_data


if __name__ == "__main__":
    #switch to correct directory
    os.chdir("../../")

    start = time.time()
    print(f"start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")

    # create PSO boxplot: achieved function values for different numbers of particles and 
    save_path = "qnn-experiments/plots/hyperparameter_plots/preliminary_test/hyperparameters_GA_DE_PSO/particle_swarm"
    create_multi_PSO_boxplot(save_path)
    # create all other hyperparameter boxplots for final experiment
    create_all_hyperparameter_boxplots()

    # create hyperparameter boxplots for preliminary tests (Dual Annealing, Genetic Algorithm, PSO, Differential Evolution)
    # for Dual Annealing: "Fehler beim Lesen der Daten: "bounds_0": "dual_annealing" is acceptable!
    create_preliminary_test_boxplots()

    end = time.time()
    print(f"end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
    print(f"total runtime (with callback): {np.round((end-start)/60,2)}min") 
    



    
    