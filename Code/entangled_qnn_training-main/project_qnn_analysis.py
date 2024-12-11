import csv
import gc
import json
import time
from datetime import datetime

import matplotlib.patches
from qnns.cuda_qnn import CudaPennylane
import re

from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from victor_thesis_experiments_main import *
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from project_qnn_sgd_for_scipy import *
from jax import jacrev
import os
import pandas as pd
from scipy.optimize import minimize, dual_annealing
import re

'''
    Analysis of final experiment for all optimizers and optimizer categories (convergence plots, boxplots, etc.).
'''

conf_ids_to_skip = [190, 191, 192, 193, 194, 210, 211, 212, 213, 214, 230, 231, 232, 233, 234]
combinations_to_skip = [["non_lin_ind","2","3"],["non_lin_ind","3","3"],["non_lin_ind","4","3"]] # Format [data_type, num_data_points, s_rank]

opt_titles = {'nelder_mead': 'Nelder-Mead', 'powell':'Powell', 'sgd':'SGD', 
              'adam':'Adam', 'rmsprop':'RMSprop', 'bfgs':'BFGS','slsqp':'SLSQP',
              'dual_annealing':'Dual Annealing','cobyla':'COBYLA',
              'genetic_algorithm':'Genetic Alg.', 'particle_swarm': 'PSO',
              'diff_evolution':'Diff. Evolution'}

datatype_list = ['random', 'orthogonal', 'non_lin_ind', 'var_s_rank']
num_data_points_list = ['1', '2', '3', '4']
s_rank_list = ['1', '2', '3', '4']
maxiter_list = [1000]
dpi=400 # change if resolution too high

mean_fun_values = pd.DataFrame(columns=["data_type", "s_rank", "num_data_points"]+list(opt_titles.keys()))

mean_fun_values_per_config = {opt:[] for opt in opt_titles.keys()}
mean_nit_values_per_config = {opt:[] for opt in opt_titles.keys()}
max_nit_values_per_config = {opt:[] for opt in opt_titles.keys()} # needed to plot correct convergence plots
mean_callback_values_per_config = {opt:{} for opt in opt_titles.keys()}

########## Data Preparation ##########
def extract_all_data_from_json_files():
    '''
        Extracts mean function and iteration values and callback values for all optimizers in final experiment.
        Prerequisite for most further analysis functions, such as convergence plots and boxplots.
    '''
    optimizers1 = ['nelder_mead', 'powell', 'sgd', 'adam', 'rmsprop', 'bfgs','slsqp','dual_annealing','cobyla']
    optimizers2 = ['genetic_algorithm', 'particle_swarm', 'diff_evolution']

    #filename = os.path.join(os.getcwd(), 'TrainLabel1.csv')
    origin_path1 = os.path.join(os.getcwd(),"qnn-experiments/optimizer_results/final_experiment_2024-10/experiment_part1")
    origin_path2 = 'qnn-experiments/optimizer_results/final_experiment_2024-10/experiment_part2_GA_PSO_DE'
    calc_mean_fun_nit_callback_values_per_config(optimizers1,origin_path1)
    calc_mean_fun_nit_callback_values_per_config(optimizers2,origin_path2)

def calc_mean_fun_nit_callback_values_per_config(opt_list, directory, max_iter=1000):
    '''
        Reads json-files from directory and computes mean function values for all optimizers in opt_list, mean number of iterations
        mean callback-values (function value history) and corresponding number of iterations (for plotting callback values in convergence plots) for each config ID. 
        Default: maximum number of itertations is 1000. 

        Fills the following dictionaries with values:
            mean_fun_values_per_config: keys are optimizers, values are a list (320 entries, one per config_id) 
            mean_nit_values_per_config: keys are optimizers, values are a list (320 entries, one per config_id) 
            max_nit_values_per_config: keys are optimizers, values are a list (320 entries, one per config_id)
            mean_callback_values_per_config: dictionary of dictionaries. Key on level 1: optimizers, 
                Key on level 2: config_id (integer), values: list (callback values)
        
        Arguments:
            opt_list (list of String): list of optimizers, names have to conform with names of optimizers in json files
            directory (String): source directory for json files
            maxiter (int, optiona): can be 100, 500 or 1000
    '''

    all_data = load_json_data(directory)
    # for each list in conf_id_list (i.e. each possible value of non-specified parameter) (and each databatch): determine a list of mean callback values
    for id in range(0,len(all_data)):
        for opt in opt_list:
            # Stepsize: Stepsize between Iterations whose fun value is saved in callback
            # for Powell, BFGS, Dual Annealing, GA and PSO: stepsize = 1 (every iteration)
            # for all other optimizers: stepsize = 10 (every 10th iteration)
            stepsize = 10
            if opt in ['powell', 'bfgs', 'dual_annealing', 'genetic_algorithm', 'particle_swarm']:
                stepsize = 1
            target_learning_rate = None
            if opt in ["sgd", "adam", "rmsprop"]:
                target_learning_rate = 0.01
            # Determine key names for maxiter
            # exceptions: Cobyla counts function evaluations instead of iterations
            maxiter_name = "maxiter"
            nit_name = "nit"
            fun_values = []
            nit_values = []
            callback_values = []
            for entry in all_data[id]:
                if isinstance(entry, dict):
                    # go through each databatch
                    for batch_key in entry:
                        if batch_key.startswith("databatch_"):
                            if opt in entry[batch_key]:
                                # get data for optimizer opt
                                batch_data = entry[batch_key][opt]
                                for key in batch_data:
                                    data = batch_data[key]            
                                    # data must be dictionary and contain keys
                                    if isinstance(data, dict):
                                        nit = data.get(nit_name, None) # nit: number of total iterations needed to reach optimal fun-value
                                        fun = data.get("fun", None) # fun: optimal fun-value reached during optimization
                                        iter = data.get(maxiter_name, None) # maxiter: number of maximum iterations optimizer was given (100, 500, or 1000)
                                        callback = data.get("callback", None) # callback: list of fun_values for every tenth iteration
                                        learning_rate = data.get("learning_rate", None) # for SGD optimizers: learning rate. Used to filter for specific learning rate. If optimizer does not use learning_rate it is None
                                        if(iter == max_iter and learning_rate == target_learning_rate): # if target_learning_rate is not specified, it is None
                                            if nit is None or opt == 'dual_annealing': #cobyla doesn't save nit and dual_annealing saves the wrong value (max_iter) for nit
                                                nit = (len(callback)-1)*stepsize
                                            if nit is not None and fun is not None:
                                                try:
                                                    nit_values.append(int(nit))
                                                    fun_values.append(float(fun))
                                                    if(callback[-1] != fun): # append optimal fun value, if it isn't already the last value in callback-list
                                                            callback.append(float(fun))
                                                    callback_values.append(callback) 
                                                except ValueError as e:
                                                    print(f"Fehler beim Konvertieren der Daten: {e}")
                                            else:
                                                print(f"Fehlende Schlüssel in den Daten: {data}")
                            else:
                                print(f"Optimierer {opt} nicht in den Datenbatch {batch_key} gefunden")
                else:
                    print("Eintrag ist kein Dictionary")
            mean_fun_values_per_config[opt].append(np.mean(fun_values))
            mean_nit_values_per_config[opt].append(np.mean(nit_values))
            # compute mean of callback fun values over each config_id and each databatch per config_id
            # for runs that used less iterations than others: fill those lists with the optimal fun value achieved
            max_len = len(max(callback_values, key=len))
            # pad right of each sublist of fun_values with optimal fun value to make it as long as the longest sublist
            for sublist in callback_values:
                opt_fun_val = sublist[-1]
                sublist[:] = sublist + [opt_fun_val] * (max_len - len(sublist))
            fun_arrays = [np.array(x) for x in callback_values]
            mean_callback_values_per_config[opt][id] = [np.mean(k) for k in zip(*fun_arrays)]
            # compute maximum of total number of iterations (nit) for each config_id and each databatch per config_id
            max_nit_values_per_config[opt].append(np.max(nit_values))
    del all_data

########## Further Info for Convergence Plots ##########
def fill_mean_fun_values():
    '''
        Computes mean function values per attribute combination (datatype, num_data_points, s_rank) of a config for each optimizer.
        Values are saved in mean_fun_values (dataframe) and 
        exported to 'qnn-experiments/plots/convergence_plots/maxiter/1000/mean_fun_values.csv'.

        Prerequisite:
            mean_fun_values_per_config is filled.

        Returns
            mean_fun_values
    '''
    if len(mean_fun_values_per_config)==0:
        raise Exception("Data has not been extracted from json files. Execute extract_all_data_from_json_files() and try again.")

    print("fill mean_fun_values")
    #prep mean_fun_values (dataframe)
    for datatype in datatype_list:
        for s_rank in s_rank_list:
            for num_data_points in num_data_points_list:
                row = [datatype, num_data_points, s_rank]
                if(row in combinations_to_skip):
                    continue
                index = ['data_type', 'num_data_points', 's_rank']
                conf_ids = get_conf_ids(datatype,num_data_points, s_rank)
                for opt in opt_titles.keys():
                    mean_fun = np.mean([mean_fun_values_per_config[opt][i] for i in conf_ids])
                    row.append(mean_fun)
                    index.append(opt)
                s = pd.Series(data=row,index=index)
                mean_fun_values.loc[len(mean_fun_values)] = s
    mean_fun_values.set_index(["data_type", "s_rank", "num_data_points"], inplace=True)
    mean_fun_values.to_csv('qnn-experiments/plots/convergence_plots/maxiter/1000/plot_info/mean_fun_values.csv', index=True)

    return mean_fun_values

def order_of_parameter_values(datatype, num_data_points, s_rank, save_path):
    '''
        Calculates the order of the variable parameter values (None-parameter) for each optimizer from best (lowest) to worst (highest)
        according to achieved optimal function value
        Two of datatype, num_data_points and s_rank are fixed and one is variable (i.e. None)

        Result is saved in save_path+"order_paramvalue_per_opt_{datatype}{num_data_points}{s_rank}.csv"

        Prerequisite:
            mean_fun_values is not empty
            exactly one of datatype, num_data_points and s_rank is None
        
        Arguments:
            datatype (String): data type of training data points ('random', 'orthogonal', 'non_lin_ind', 'var_s_rank' or None)
            num_data_points (String): number of training data points (1, 2, 3 or 4 or None)
            s_rank (String): Schmidt Rank of training data (1, 2, 3 or 4 or None)
            save_path (String): file path for csv file
        
        returns:
            content of csv file as Panda Dataframe
    '''
    # create correct directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if len(mean_fun_values)==0:
        raise Exception("Data has not been extracted from json files. Execute extract_all_data_from_json_files() and try again.")
    
    # check that only one parameter (data_type, num_data_points, s_rank) is None:
    param_names = ['data_type', 'num_data_points', 's_rank']
    params = [datatype, num_data_points, s_rank]
    not_none_indices = [i for i in range(len(params)) if params[i] != None]
    if(len(not_none_indices) != 2):
        raise Exception('Exactly one parameter of data_type, num_data_points and s_rank must be None')
    
    # determine the non-variable parameters
    param0 = param_names[not_none_indices[0]]
    param1 = param_names[not_none_indices[1]]
    m0 = (mean_fun_values.index.get_level_values(param0) == params[not_none_indices[0]])
    m1 = (mean_fun_values.index.get_level_values(param1) == params[not_none_indices[1]])
    df = mean_fun_values[m0 & m1]

    # determine variable parameter
    none_index = [i for i in range(len(params)) if params[i] == None]
    var_param = param_names[none_index[0]]

    df1 = (df.reset_index().drop([param0, param1], axis=1).set_index([var_param])
           .apply(lambda x: x.index[x.argsort()])
           .reset_index(drop=True)
           )

    df1.to_csv(save_path+f"order_paramvalue_per_opt_{datatype}{num_data_points}{s_rank}.csv")
    return df1

def best_opt_per_param_combination(save_path):
    '''
        saves the order of optimizers for each parameter combination (datatype, num_data_points, s_rank) from best to worst
        and the achieved mean optimal function value.
        Result is saved as a txt-file in save_path

        Prerequisite:
            mean_fun_values is not empty
        
        Arguments:
            save_path (String): file path for csv files
        
        Returns:
            contents of csv files as Pandas Dataframes
    '''
    if len(mean_fun_values)==0:
        raise Exception("Data has not been extracted from json files. Execute extract_all_data_from_json_files() and try again.")

    #determine order of optimizers
    df1 = (mean_fun_values[list(opt_titles.keys())]
           .apply(lambda x: x.sort_values().index.tolist(), axis=1, result_type='expand')
           .pipe(lambda x: x.set_axis(x.columns+1, axis=1))
           .reset_index()
          )
    df1.to_csv(save_path+"best_opt_per_param_combination.csv")

    #determine median and standard deviation (and other characteristics) of achieved values 
    df2 = (mean_fun_values[list(opt_titles.keys())]
           .astype(float)
           .transpose()
           .describe(include='all')
           .transpose()
           .reset_index()
           )
    df2.to_csv(save_path+"best_opt_per_param_combination_mean_std.csv")

    return df1, df2
                         
def compute_deltas(datatype, num_data_points, s_rank, save_path):
    '''
        Calculates the achieved function values and the difference between each achieved function value for each value of the variable parameter.
        Two of datatype, num_data_points and s_rank are fixed
        and one is variable (None)

        Results are saved in csv files:
            .../mean_fun_values_{datatype}{num_data_points}{s_rank}.csv
            .../delta_mean_fun_values_{datatype}{num_data_points}{s_rank}.csv

        Prerequisite:
            mean_fun_values is not empty
            exactly one of datatype, num_data_points and s_rank is None

        Arguments:
            datatype (String): data type of training data points ('random', 'orthogonal', 'non_lin_ind', 'var_s_rank' or None)
            num_data_points (String): number of training data points (1, 2, 3 or 4 or None)
            s_rank (String): Schmidt Rank of training data (1, 2, 3 or 4 or None)
            save_path (String): file path for csv file
        
        returns:
            content of csv files as Panda Dataframes      
        
    '''
    # create correct directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if len(mean_fun_values)==0:
        raise Exception("Data has not been extracted from json files. Execute extract_all_data_from_json_files() and try again.")
    
    # check that only one parameter (data_type, num_data_points, s_rank) is None:
    param_names = ['data_type', 'num_data_points', 's_rank']
    params = [datatype, num_data_points, s_rank]
    not_none_indices = [i for i in range(len(params)) if params[i] != None]
    if(len(not_none_indices) != 2):
        raise Exception('Exactly one parameter of data_type, num_data_points and s_rank must be None')
    
    # determine the non-variable parameters
    param0 = param_names[not_none_indices[0]]
    param1 = param_names[not_none_indices[1]]
    m0 = (mean_fun_values.index.get_level_values(param0) == params[not_none_indices[0]])
    m1 = (mean_fun_values.index.get_level_values(param1) == params[not_none_indices[1]])
    df = mean_fun_values[m0 & m1]
    df_delta = df.diff()
    df_relative_delta = df.pct_change()

    df.to_csv(save_path+f"/mean_fun_values_{datatype}{num_data_points}{s_rank}.csv")
    df_delta.to_csv(save_path+f"/delta_mean_fun_values_{datatype}{num_data_points}{s_rank}.csv")
    df_relative_delta.to_csv(save_path+f"/rel_delta_mean_fun_values_{datatype}{num_data_points}{s_rank}.csv")

    return df, df_delta

def compute_convergence_plot_information():
    '''
        Computes information pertaining to the convergence plots, 
        such as deltas between every achieved function value per variable parameter (datatype, s_rank of num_data_points),
        average delta per variable parameter (see compute_deltas() for both)
        optimizer with best results per combination of parameters (see best_opt_per_param_combination())
        order of parameter values (variable parameter) from best to worst results (for convergence plots where order is not visible in plot) (see order_of_parameter_values())

        Requirement: mean_fun_values is not empty

        Result is saved in a txt file in each convergence plot folder
    '''
    # compute deltas

    save_path='qnn-experiments/plots/convergence_plots/maxiter/1000/plot_info/'
    
    # create correct directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    best_opt_per_param_combination(save_path)
    df_deltas_list = []

    maxiter = 1000
    for datatype in datatype_list:
        print(f"datatype: {datatype}")
        # convergence plot information for variable s_rank, but fixed datatype and num_data_points
        print("Variable S-Rank in progress...")
        for num_data_points in num_data_points_list:
            print(f"num_data_points: {num_data_points}")
            path = save_path+f'datatype/{datatype}/num_data_points/' 
            df, df_deltas = compute_deltas(datatype, num_data_points, None, path)
            df_deltas_list.append(df_deltas)
            order_of_parameter_values(datatype, num_data_points, None, path)
    # more info about deltas

    df_deltas_list = []
    for datatype in datatype_list:       
        # convergence plot information for variable num_data_points, but fixed datatype and s_rank
        print("Variable number of data points in progress...")
        for s_rank in s_rank_list:
            if(s_rank==3 and datatype=="non_lin_ind"):
                print("skipping s-rank = 3, datatype = non_lin_ind")
                continue
            print(f"s-rank: {s_rank}")
            path = save_path+f'datatype/{datatype}/s_rank/'
            compute_deltas(datatype, None, s_rank, path)
            order_of_parameter_values(datatype, None, s_rank, path)

    df_deltas_list = []  
    # convergence plot information for variabel datatype, but fixed num_data_points and s_rank
    print("Variable datatype in progress...")
    for s_rank in s_rank_list:
        print(f"s-rank: {s_rank}")
        for num_data_points in num_data_points_list:
            print(f"num_data_points: {num_data_points}")
            path = save_path+f's_rank/{s_rank}/num_data_points/'
            compute_deltas(None, num_data_points, s_rank, path)
            order_of_parameter_values(None, num_data_points, s_rank, path)

########## Helper Functions ##########

def get_conf_ids(data_type, num_data_points, s_rank,every_fifth_config=False):
    '''
        Returns a list of config ids that correspond to the combination of values for data_type, num_data_points and s_rank.
        
        Arguments:
            datatype (String): data type of training data points ('random', 'orthogonal', 'non_lin_ind', 'var_s_rank')
            num_data_points (String): number of training data points (1, 2, 3 or 4)
            s_rank (String): Schmidt Rank of training data (1, 2, 3 or 4)
            every_fifth_config (bool): true, if there are only json files for every fifth config (used for testing purposes). Default: false

        Returns:
            conf_id_list (list of int): list of corresponding config_ids
    '''
    # use to determine which conf_ids should be added to list (every single one or every fifth)
    mod = 1
    if(every_fifth_config):
        mod=5
    data = []
    conf_id_list = []
    file_path = "Code/entangled_qnn_training-main/data/configDict.json"
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            for i in range(len(data)):
                if(i%mod == 0 and data[str(i)]["data_type"]==data_type and data[str(i)]["num_data_points"]==num_data_points and data[str(i)]["s_rank"]==s_rank):
                    conf_id_list.append(i)
        except json.JSONDecodeError:
            print(f"Fehler beim Laden der Datei: {file_path}")
    return(conf_id_list)

def get_conf_ids_forParam(distinctionParam, paramValue):
    '''
        Adds all configs that match with a specified parameter Value for all other parameters staying unspecified
        
        Arguments:
            disctionctionParam (String): one of data_type, num_data_points, s_rank
            paramValue (String): value of distinctionParam
        
        Returns:
            conf_id_list (list of int): list of corresponding config_ids
    '''
    
    data = []
    conf_id_list = []
    file_path = "Code/entangled_qnn_training-main/data/configDict.json"
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            for i in range(len(data)):
                if(data[str(i)][distinctionParam]==paramValue):
                    conf_id_list.append(i)
        except json.JSONDecodeError:
            print(f"Fehler beim Laden der Datei: {file_path}")
    return(conf_id_list)

def load_json_data(directory, conf_id_list=range(0,320)):
    '''
        Load JSON-data for each config_id in conf_id_list from files saved in directory.
        File names start with "conf_{config_id}_" and end with ".json".
        Default conf_id_list is all configs, i.e. 0 to 319.

        Arguments:
            directory (String): source directory for JSON files
            conf_id_list (list of int, optional): list of config_ids 

        Returns:
            all_data (dict): dict where keys are ids and values are a list of all corresponding json-files loaded as dictionaries.
    '''
    all_data = {}
    for id in conf_id_list:
        all_data[id] = []
        for filename in os.listdir(directory):
            if filename.endswith('.json') and filename.startswith(f'conf_{id}_'):
                file_path = os.path.join(directory, filename)
                #print(f"Lade Datei: {file_path}")
                with open(file_path, 'r') as file:
                    try:
                        all_data[id].append(json.load(file))
                    except json.JSONDecodeError:
                        print(f"Fehler beim Laden der Datei: {file_path}")
        if not all_data:
            print("Keine JSON-Dateien gefunden oder alle Dateien sind fehlerhaft.")
    return all_data

########## Boxplots Plots for bounds testing ##########

def extract_solution_x_data(json_data):
    '''
        Extract mean x_min and x_max value for every optimizer for every bound for one config.
        Needed for create_min_max_boxplots
        
        Arguments:
            json_data (dict): Dictionary containing all data from json files, first level keys are config_ids
        Returns:
            res_min (dict): all smallest solution x-values per optimizer and bounds value
            res_max (dict): all largest solution x-values per optimizer and bounds value
            res_min_max (dict): the smallest and largest solution x-value per optimizer
    '''
    gradient_free = ["nelder_mead", "powell", "cobyla"]
    gradient_based = ["sgd", "adam", "rmsprop", "bfgs", "dual_annealing","slsqp"]
    optimizers = gradient_based + gradient_free
    bounds_batches = ["bounds_0", "bounds_1", "bounds_2", "bounds_3", "bounds_4"]
    databatches = ["databatch_0", "databatch_1", "databatch_2", "databatch_3", "databatch_4"]

    # prepare results dict
    # bounds_i : opt_1 : [(x_min1, x_max1), (x_min2, x_max2),...], opt_2 : ...
    res_min = {"bounds_0": {}, "bounds_1": {}, "bounds_2": {}, "bounds_3": {}, "bounds_4": {}}
    res_max = {"bounds_0": {}, "bounds_1": {}, "bounds_2": {}, "bounds_3": {}, "bounds_4": {}}

    for i in range(len(json_data)):
        print(f"Verarbeite config_{i}")
        for databatch_id in databatches:
            print(f"Verarbeite {databatch_id}")
            for bounds_id in bounds_batches:
                print(f"Verarbeite {bounds_id}")
                for opt in optimizers:
                    print(f"Verarbeite {opt}")
                    try:
                        dict = json_data[i][0][databatch_id][bounds_id][opt]["0"]
                        if opt not in res_min[bounds_id]:
                            res_min[bounds_id][opt] = []
                        if opt not in res_max[bounds_id]:
                            res_max[bounds_id][opt] = []
                        x = dict["x"]
                        x = [float(idx) for idx in x.strip('[ ]').split()]
                        res_min[bounds_id][opt].append(np.min(x))
                        res_max[bounds_id][opt].append(np.max(x))
                    except KeyError as e:
                        print(f"Fehler beim Lesen der Daten: {e}")

    # Berechne Pro Optimierer (pro Bounds) untere x-Grenze und obere x-Grenze
    res_min_max = {"bounds_0": {}, "bounds_1": {}, "bounds_2": {}, "bounds_3": {}, "bounds_4": {}}
    for bounds_id in bounds_batches:
        print(f"Verarbeite {bounds_id}")
        for opt in optimizers:
            print(f"Verarbeite {opt}")
            try:
                res_min_max[bounds_id][opt] = (np.min(res_min[bounds_id][opt]),np.max(res_max[bounds_id][opt]))
            except KeyError:
                print(f'Optimierer existiert für diese bounds nicht.')
    
    return res_min, res_max, res_min_max

def create_min_max_boxplots(res_min, res_max, save_path):
    '''
        Creates several boxplot where minimal and maximal solution x-values are plotted per optimizer. One plot per bounds value.
        Save path for plots: savepath+'{bounds_id}_boxplot_no_outliers.png'

        Prerequisite: 
            exactly one of datatype, num_data_points and s_rank is None

        Arguments:
            res_min (dict): first level keys are values for bounds ("bounds_0", "bounds_1", ..., "bounds_4") and values are list of lowest function values
            res_max (dict): first level keys are values for bounds ("bounds_0", "bounds_1", ..., "bounds_4") and values are list of highest function values
            save_path (String): save path for plots
    '''
    bounds = {"bounds_0": "No Bounds", "bounds_1": r"$[0, 2\pi]$", "bounds_2": r"$[0, 4\pi]$", "bounds_3": r"$[-2\pi, 2\pi]$", "bounds_4": r"$[-4\pi, 4\pi]$"}
    bounds_limits = {"bounds_1": [0, 2*np.pi], "bounds_2": [0, 4*np.pi], "bounds_3": [-2*np.pi, 2*np.pi], "bounds_4": [-4*np.pi, 4*np.pi]}
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for bounds_id in bounds.keys():
        plt.figure(figsize=(9.2,6))
        # plot vertical grey lines of bounds, for "no bounds" plot [0, 2*pi]
        if(bounds_id in bounds_limits.keys()):
            interval = bounds_limits[bounds_id]
            for x in interval:
                plt.axvline(x,c="grey")
        else:
            interval = bounds_limits["bounds_1"]
            for x in interval:
                plt.axvline(x,c="grey")
        data_min = res_min[bounds_id]
        data_max = res_max[bounds_id]
        x = np.array([(i+1)*1000 for i in range(len(data_min.keys()))])
        # adapt colors
        c1 = list(matplotlib.colors.to_rgba("darkseagreen"))
        c1[3] = 0.5 # make more transparent
        c1 = tuple(c1)
        c2 = list(matplotlib.colors.to_rgba("skyblue"))
        c2[3] = 0.5 # make more transparent
        c2 = tuple(c2)
        plt.boxplot(data_min.values(), sym="", patch_artist=True, boxprops=dict(facecolor=c1,hatch='oo'), medianprops=dict(linewidth=2), vert=False,positions=x-100,widths=220)
        plt.boxplot(data_max.values(), sym="", patch_artist=True, boxprops=dict(facecolor=c2,hatch='//'), medianprops=dict(linewidth=2), vert=False,positions=x+100,widths=220)
        
        # legend
        dg_patch = matplotlib.patches.Patch(facecolor=c1,hatch='o', label='minimal x values')
        blue_patch = matplotlib.patches.Patch(facecolor=c2,hatch=r'//', label='maximal x values')
        plt.legend(handles=[dg_patch,blue_patch], labelspacing=1, handlelength=2, fontsize=10) 
        plt.xticks(fontsize=12)
        if(bounds_id == "bounds_0"):
            plt.yticks(ticks=x,labels=data_min.keys(),fontsize=12)
        else:
            plt.yticks(ticks=x,labels=data_min.keys(),fontsize=10)
        plt.ylabel('Optimizer',fontsize=14)
        plt.xlabel('Minimal (lower) and maximal (upper) x-values',fontsize=14)
        plt.title(f"Minimal and Maximal x-Values for bounds: {bounds[bounds_id]}",fontsize=20)
        plt.grid(True)
        file_path = os.path.join(save_path, f'{bounds_id}_boxplot_no_outliers.png')
        plt.savefig(file_path, dpi=dpi)
        plt.close()

def make_bounds_boxplots():
    '''
        Makes boxplots for all bounds values. 
        savepath: 'qnn-experiments/plots/hyperparameter_plots/preliminary_test/bounds'
    '''
    directory = "qnn-experiments/optimizer_results/bounds_2024-07-29"
    save_path = f'qnn-experiments/plots/hyperparameter_plots/preliminary_test/bounds'

    json_data = load_json_data(directory)
    res_min, res_max, res_min_max = extract_solution_x_data(json_data)
    create_min_max_boxplots(res_min, res_max, save_path)

########## Convergence Plots ##########

def convergence_plot_per_optimizer(save_path, mean_fun_data, mean_nit_data, opt, maxiter, data_type, num_data_points, s_rank, learning_rate=None):
    '''
        Convergence plot for mean callback values where exactly one parameter of data_type, num_data_points or s_rank is None and thus variable.
        mean_fun_data is a dictionary where the possible values for the variable parameter are the key and each value saved for a key is a list of fun_values
        mean_nit_data is a list of the corresponding number of iterations for the found optimal fun value (last value in each list in mean_fun_data)

        Arguments:
            save_path (String): save path for plot
            mean_fun_data (dict of lists): keys are values for the parameter (of data_type, num_data_points or s_rank) that is None, values are list of function values
            mean_nit_data (dict of lists): keys are values for the parameter (of data_type, num_data_points or s_rank) that is None, values is maximal number of iterations for this value
            opt (String): optimizer name
            maxiter (int): maximal number of iterations, for plot title
            datatype (String): random, orthogonal, non_lin_ind, var_s_rank or None
            num_data_points (String): 1,2,3,4 or None
            s_rank (String): 1,2,3,4 or None 
            learning_rate (Boolean, optional): true if data is restricted to learning_rate = 0.01
    '''
    # create correct directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Stepsize: Stepsize between Iterations whose fun value is saved in callback (influences x-axis of plot)
    # for Powell, BFGS, Dual Annealing, GA, PSO and DE: stepsize = 1 (every iteration)
    # for all other optimizers: stepsize = 10 (every 10th iteration)
    stepsize = 10
    if opt in ['powell', 'bfgs', 'dual_annealing', 'genetic_algorithm', 'particle_swarm']:
        stepsize = 1
    
    #determine what parameter is variable (i.e. None in argument list) and check that only one parameter is None
    param_names = ['data_type', 'num_data_points', 's_rank']
    params = [data_type, num_data_points, s_rank]
    none_indices = [i for i in range(len(params)) if params[i] == None]
    if(len(none_indices)>1):
        raise Exception('Only one parameter of data_type, num_data_points and s_rank is allowed to be None')
    none_param = param_names[none_indices[0]]

    # Create title: Only add parameters that are not variable (i.e. None) & add learning rate for SGD optimizers if applicable
    title = f'Convergence plot for {opt_titles[opt]}, maxiter = {maxiter}, '
    if(opt in ['sgd', 'adam', 'rmsprop'] and learning_rate is not None):
        title += f'learning rate = {learning_rate},'
    title += '\n'
    param_titles = {'data_type': "Data Type", 'num_data_points': "Number of Data Points", 's_rank': "Schmidt Rank"}
    j=0
    for i in range(0,3):
        if i not in none_indices:
            title += f"{param_titles[param_names[i]]}: {params[i]}"
            j+=1
            if j < 2:
                title += ", "
    
    #colors for each config id
    #cmap = matplotlib.colormaps["tab10"]
    cmap = ['skyblue', 'darkseagreen', 'green', 'grey']
    plt.figure(figsize=(7,5.3))
    c = 0 # needed to determine correct color
    for param_value in mean_fun_data.keys():
        #color = cmap(c/4) #use when loading a colormap from matplotplib
        color = cmap[c]
        label = f"{none_param} = {param_value}"
        y = mean_fun_data[param_value]
        # Genetic Algorithm saves callback function values for all maxiter iterations, instead of only nit iterations
        # hence max_nit_value = maxiter for Genetic Algorithm
        if opt == "genetic_algorithm":
            x = np.arange(0,len(y)*stepsize,stepsize)
        else:
            x = np.append(np.arange(0,(len(y)-1)*stepsize, stepsize), mean_nit_data[param_value])
        if x[-1] < x[-2]:
            print("achtung: plot problem:", opt, x[-1], x[-2])
        plt.plot(x,y, color=color, label=label)
        c += 1
    plt.ylim(0,1)
    plt.xlabel('Iteration',fontsize=18)
    plt.ylabel('Function value',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    if(opt in ['sgd', 'adam', 'rmsprop']):
        plt.title(title,fontsize=13)
    else:
        plt.title(title,fontsize=16)
    plt.grid(True)
    file_path = os.path.join(save_path, f'{opt}_convergence_fun_{data_type}{num_data_points}{s_rank}.png')
    plt.savefig(file_path, dpi=dpi)
    plt.close()

def calc_convergence_data(datatype, num_data_points, s_rank):
    '''
        Calculates the mean of all configurations that fit 
        Exactly one of data_type, num_data_points or s_rank is None, hence callback data for all possible values of this parameter will be extracted.
        The mean achieved function value history per optimizer per value of the None-parameter is calculated for every optimizer.

        Needed for make_all_convergence_plots()
        
        Prerequisite: 
            mean_callback_values_per_config is not empty.
            exactly one of datatype, num_data_points and s_rank is None

        Arguments:
            datatype (String): random, orthogonal, non_lin_ind, var_s_rank or None
            num_data_points (String): 1,2,3,4 or None
            s_rank (String): 1,2,3,4 or None
        
        Results: 
            callback_values: dictionary of dictionaries. Level 1 key: optmizer, level 2 key: value of None-parameter, values: List of callback values
            nit_values: dictionary of dictionaries. Level 1 key: optmizer, level 2 key: value of None-parameter, values: corresponding maximum number of iterations for this list of callback values
    '''
    if len(mean_callback_values_per_config)==0:
        raise Exception("Data has not been extracted from json files. Execute extract_all_data_from_json_files() and try again.")

    callback_values = {}
    nit_values = {}

    # check that only one parameter (data_type, num_data_points, s_rank) is None:
    param_values = {'data_type': ["random", "orthogonal", "var_s_rank", "non_lin_ind"], 'num_data_points': ["1","2","3","4"], 's_rank': ["1","2","3","4"]}
    param_names = ['data_type', 'num_data_points', 's_rank']
    params = [datatype, num_data_points, s_rank]
    none_indices = [i for i in range(len(params)) if params[i] == None]
    if(len(none_indices)>1):
        raise Exception('Only one parameter of data_type, num_data_points and s_rank is allowed to be None')
    
    # determine the variable parameter (None-parameter)
    none_param = param_names[none_indices[0]]
    
    for opt in opt_titles.keys():
        callback_values[opt] = {}
        nit_values[opt] = {}
        for value in param_values[none_param]:
            params_current = [value if v is None else v for v in params] #determine correct parameter values (substitute value for None)
            if params_current in combinations_to_skip:
                continue
            # determine list of config-Ids that belong to this specific combinations of parameters
            conf_id_list = get_conf_ids(data_type=params_current[0], num_data_points=params_current[1], s_rank=params_current[2])
            callbacks = [mean_callback_values_per_config[opt][i] for i in conf_id_list]
            
            # compute mean of callback fun values over each config_id 
            # for conf_ids that needed less iterations than others: fill those lists with the optimal fun value achieved
            max_len = len(max(callbacks, key=len))
            # pad right of each sublist of fun_values with optimal fun value to make it as long as the longest sublist
            for sublist in callbacks:
                opt_fun_val = sublist[-1]
                sublist[:] = sublist + [opt_fun_val] * (max_len - len(sublist))
            fun_arrays = [np.array(x) for x in callbacks]
            callback_values[opt][value] = [np.mean(k) for k in zip(*fun_arrays)]

            # compute correct maximum number of iterations
            nits = [max_nit_values_per_config[opt][i] for i in conf_id_list]
            nit_values[opt][value] = np.max(nits)
    return callback_values, nit_values

def make_all_convergence_plots(save_path='qnn-experiments/plots/convergence_plots/'):
    '''
        Make all convergence plots for each combination of configuration attributes (data type, number of data points and Schmidt-rank)
        for each optimizer in optimizers for maximum number of iterations 100 and 1000. 
        Plots are saved as .png files in qnn-experiments/plots/convergence_plots.
        Example file path for a plot with maximum 1000 iterations, datatype=random and num_data_points=1:
        qnn-experiments/plots/convergence_plots/maxiter/1000/datatype/random/num_data_points/

        Pre-Req:
            if a save_path is given it must end with "/"
            mean_callback_values_per_config is not empty.
        
        Arguments:
            save_path (String, optional): save path for plots
    '''
    
    for maxiter in maxiter_list:
        print(f"maxiter: {maxiter}")
        for datatype in datatype_list:
            print(f"datatype: {datatype}")
            # convergence plots for variable s_rank, but fixed datatype and num_data_points
            print("Variable S-Rank in progress...")
            for num_data_points in num_data_points_list:
                print(f"num_data_points: {num_data_points}")
                path = save_path+f'maxiter/{maxiter}/datatype/{datatype}/num_data_points/{num_data_points}'
                callback_values, nit_values = calc_convergence_data(datatype, num_data_points, None)
                for opt in opt_titles.keys():
                    learning_rate = None
                    if opt in ["sgd", "adam", "rmsprop"]:
                        learning_rate = 0.01
                    convergence_plot_per_optimizer(path, callback_values[opt],nit_values[opt], opt, maxiter, datatype, num_data_points, None, learning_rate)
                    print(f"optimizer: {opt} done")
            
            # convergence plots for variable num_data_points, but fixed datatype and s_rank
            print("Variable number of data points in progress...")
            for s_rank in s_rank_list:
                if(s_rank==3 and datatype=="non_lin_ind"):
                    print("skipping s-rank = 3, datatype = non_lin_ind")
                    continue
                print(f"s-rank: {s_rank}")
                path = save_path+f'maxiter/{maxiter}/datatype/{datatype}/s_rank/{s_rank}'
                callback_values, nit_values = calc_convergence_data(datatype, None, s_rank)
                for opt in opt_titles.keys():
                    learning_rate = None
                    if opt in ["sgd", "adam", "rmsprop"]:
                        learning_rate = 0.01
                    convergence_plot_per_optimizer(path, callback_values[opt],nit_values[opt], opt, maxiter, datatype, None, s_rank, learning_rate)
                    print(f"optimizer: {opt} done") 
        
        # convergence plots for variabel datatype, but fixed num_data_points and s_rank
        print("Variable datatype in progress...")
        for s_rank in s_rank_list:
            print(f"s-rank: {s_rank}")
            for num_data_points in num_data_points_list:
                print(f"num_data_points: {num_data_points}")
                path = save_path+f'maxiter/{maxiter}/s_rank/{s_rank}/num_data_points/{num_data_points}'
                callback_values, nit_values = calc_convergence_data(None, num_data_points, s_rank)
                for opt in opt_titles.keys():
                    learning_rate = None
                    if opt in ['sgd', 'adam', 'rmsprop']:
                        learning_rate = 0.01 
                    convergence_plot_per_optimizer(path, callback_values[opt],nit_values[opt], opt, maxiter, None, num_data_points, s_rank, learning_rate)
                    print(f"optimizer: {opt} done")

########## Category Boxplots for Optimizer Categories ##########

def plot_boxplots(boxplot_save_path, title,data_GradFree,data_EVO,data_GradBased,xAxisName,iterList,labels=['Gradient-Free','Evolution-Based','Gradient-Based']):
    '''
        Plot 3 boxplots (values in data_GradFree, data_EVO and data_GradBased) per value of xAxisName 
        (values for xAxisName are given in iterList). 

        Arguments:
            boxplot_save_path (String): save path for plot
            title (String): title of plot
            data_GradFree,data_EVO,data_GradBased (dict of lists): contains achieved function values for each of the three boxplots for each value of xAxisName
            xAxisName (String): variable parameter (x Axis of plot)
            iterList (list): values for x Axis
            labels (list, optional): names of each of the three boxplots, default: ['Gradient-Free','Evolution-Based','Gradient-Based']
    '''
    #data_GradFree={1:[0.2,0.8],2:[0.2,0.8],3:[0.2,0.8],4:[0.2,0.8]} 
    #data_GradBased={1:[0.2,0.4],2:[0.2,0.4],3:[0.2,0.4],4:[0.2,0.4]} 
    #data_EVO={1:[0.2,0.4],2:[0.2,0.4],3:[0.2,0.4],4:[0.2,0.4]} 
    if not os.path.exists(boxplot_save_path):
        os.makedirs(boxplot_save_path)

    
    fig, ax = plt.subplots(figsize=(10, 6))   
    #plt.figure()
    
    #labels=list(data_GradFree.keys())
    if(xAxisName=='maxiter'):
        positions1=[0.80, 2.10, 3.40]
        positions2=[1.20, 2.50, 3.80]
        positions3=[1.60, 2.90, 4.20]
        widths=[0.25,0.25,0.25]
        x=positions2
    else:
        positions1=[0.80, 2.10,    3.40, 4.70]
        positions2=[1.20, 2.50, 3.80, 5.10]
        positions3=[1.60, 2.90, 4.20,5.50]
        widths=[0.25,0.25,0.25,0.25]
        x=positions2

    valuesGradFree = [data_GradFree[key] for key in data_GradFree.keys()]
    values_EVO = [data_EVO[key] for key in data_EVO.keys()]
    valuesGradBased = [data_GradBased[key] for key in data_GradBased.keys()]
    
    # adapt colors:
    c = []
    color_names = ["darkseagreen", "green", "skyblue"]
    for color in color_names:
        c_temp = list(matplotlib.colors.to_rgba(color))
        c_temp[3] = 0.5 # make more transparent
        c_temp = tuple(c_temp)
        c.append(c_temp)

    bp1=plt.boxplot(valuesGradFree,positions=positions1, patch_artist=True, boxprops=dict(facecolor=c[0],hatch='oo'), medianprops=dict(linewidth=2), manage_ticks=True ,widths=widths )
    bp2=plt.boxplot(values_EVO,positions=positions2, patch_artist=True, boxprops=dict(facecolor=c[1],hatch='xx'), medianprops=dict(linewidth=2), manage_ticks=True ,widths=widths)
    bp3=plt.boxplot(valuesGradBased,positions=positions3, patch_artist=True, boxprops=dict(facecolor=c[2],hatch='//'), medianprops=dict(linewidth=2), manage_ticks=True ,widths=widths)
    bps=[bp1,bp2,bp3]
    
    plt.xticks(ticks=x, labels=iterList)
    dg_patch = matplotlib.patches.Patch(facecolor=c[0],hatch='o', label=labels[0])
    green_patch =matplotlib.patches.Patch(facecolor=c[1],hatch='x', label=labels[1])

    blue_patch = matplotlib.patches.Patch(facecolor=c[2],hatch=r'//', label=labels[2])
    leg=plt.legend(handles=[dg_patch,green_patch,blue_patch], labelspacing=1, handlelength=3) 

    for patch in leg.get_patches():
        patch.set_height(15)
        patch.set_y(-2)
        
    #ax.set_xticklabels(labels)
    plt.xlabel(xAxisName)
    plt.ylabel('Function Value (fun)')
    plt.title(title)
    #plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    file_path = os.path.join(boxplot_save_path, title+'_bps.png')
    fig.savefig(file_path, dpi=dpi)

    plt.close()

def extract_func_data(directory, conf_id_list,every_fifth_config=False,target_learning_rate=None):
    '''
        For each entry in json_data (each configuration) extract list of 
        gradientfree and gradientbased optimizers reached final functional values for specified conf_id list

        Arguments:
            directory (String)
            conf_id_list (list of int)
            every_fifth_config (Boolean, optional): true if every fifth configuration is 
            target_learning_rate (Boolean, optional): true if learning rate for SGD optimizers is to be restricted to only 0.01
   
    '''
    # Determine key names for maxiter
    maxiter_name = "maxiter"
    nit_name = "nit"

    gradient_free_original = ["nelder_mead", "powell", "cobyla", "dual_annealing"]
    gradient_free_evolutional = ["genetic_algorithm", "particle_swarm", "diff_evolution"]

    #   wihout SGD!
    gradient_based = ["adam", "rmsprop", "bfgs","slsqp"]
    


    # for each list in conf_id_list (i.e. each possible value of non-specified parameter) (and each databatch): determine a list of mean callback values
    fun_values_grad = []
    fun_values_gradfree_og = []
    fun_values_gradfree_evolutional = []
    print(conf_id_list)
    
    all_data = load_json_data(directory, conf_id_list)
    fun_values = []
    #if(config_ids_must_be_skipped(data_type,num_data_points,s_rank,value)):
    #    continue
    for id in conf_id_list:
        if id not in conf_ids_to_skip:
            for entry in all_data[id]:
                if isinstance(entry, dict):
                    # go through each databatch
                    for batch_key in entry:
                        if batch_key.startswith("databatch_"):
                            for opt in gradient_free_original:
                                if opt in entry[batch_key]:
                                    # get data for optimizer opt
                                    batch_data = entry[batch_key][opt]
                                    for key in batch_data:
                                        data = batch_data[key]
                                        
                                        # data must be dictionary and contain keys
                                        if isinstance(data, dict):
                                            fun = data.get("fun", None) # fun: optimal fun-value reached during optimization
                                            fun_converted=float(fun)
                                            fun_values_gradfree_og.append(fun_converted)

                            for opt in gradient_free_evolutional:
                                if opt in entry[batch_key]:
                                    # get data for optimizer opt
                                    batch_data = entry[batch_key][opt]
                                    for key in batch_data:
                                        data = batch_data[key]
                                        
                                        # data must be dictionary and contain keys
                                        if isinstance(data, dict):
                                            fun = data.get("fun", None) # fun: optimal fun-value reached during optimization
                                            fun_converted=float(fun)
                                            fun_values_gradfree_evolutional.append(fun_converted)

                            for opt in gradient_based:
                                if opt in entry[batch_key]:
                                    # get data for optimizer opt
                                    

                                    batch_data = entry[batch_key][opt]
                                    for key in batch_data:
                                        data = batch_data[key]
                                            
                                        # data must be dictionary and contain keys
                                        if isinstance(data, dict):
                                            fun = data.get("fun", None) # fun: optimal fun-value reached during optimization
                                            fun_converted=float(fun)
                                            if opt in ["sgd", "adam", "slsqp"]:
                                                if data.get("learning_rate", None)==0.01:
                                                    fun_values_grad.append(fun_converted) 
                                            else:
                                                fun_values_grad.append(fun_converted) 

    return fun_values_gradfree_og, fun_values_gradfree_evolutional, fun_values_grad
    
def makeCategoryBoxplots(xAxisName):
    '''
        Making boxplots for all three optimizer types (gradient-based, gradient-free and evolution-based) 
        given one of data_type, num_data_points or Schmidt rank (specified in xAxisName)

        plots are saved in 'qnn-experiments/plots/category_plots/three_categories_withoutSGD/'

        Arguments:
            xAxisName (String): variable training data attribute, i.e. x-axis of plot (data_type, num_data_points or s_rank)

    '''
    if(xAxisName=='data_type'):
        iterList = ['random', 'orthogonal', 'non_lin_ind', 'var_s_rank']
    elif(xAxisName=='num_data_points'):
        iterList = ['1', '2', '3', '4']
    elif(xAxisName=='s_rank'):
        iterList = ['1', '2', '3', '4']
    #elif(xAxisName=='maxiter'):
    #    iterList = [100,1000]

    print("process all...")
    #for maxiter in maxiter_list:
    #    print(f"maxiter: {maxiter}")
    #    path = save_path+'maxiter'
    
    origin_path = 'qnn-experiments/optimizer_results/final_experiment_2024-10/experiment_part1'
    origin_path_2nd = 'qnn-experiments/optimizer_results/final_experiment_2024-10/experiment_part2_GA_PSO_DE'
    save_path='qnn-experiments/plots/category_plots/three_categories_withoutSGD/'
    r_rankDict_grad={}
    r_rankDict_gradFree={}
    r_rankDict_gradFree_EVO={}

    count=1
    for iter in iterList:
        print(f"s_rank: {iter}")
        #path = save_path+'s_rank'
        #seperate handeling for maxiteration
        confIdList=get_conf_ids_forParam(xAxisName, iter)
        gradFreeList, gradFreeEvoList, gradBasedList=extract_func_data(origin_path,confIdList)
        gradFreeList_2nd,gradFreeEvoList_2nd, gradBasedList2nd=extract_func_data(origin_path_2nd,confIdList)

        print(gradFreeEvoList)        
        for entryTemp in gradFreeList_2nd:
            gradFreeList.append(entryTemp)
        #for entryTemp in gradFreeList_2nd: 
        #    gradBasedList.append(gradBasedList2nd)

        r_rankDict_gradFree[count]=gradFreeList
        r_rankDict_gradFree_EVO[count]=gradFreeEvoList_2nd
        r_rankDict_grad[count]=gradBasedList
        count=count+1
    plot_boxplots( save_path,xAxisName+'_withoutSGD', data_GradFree=r_rankDict_gradFree, data_EVO=r_rankDict_gradFree_EVO, data_GradBased=r_rankDict_grad, xAxisName=xAxisName,iterList=iterList)


if __name__ == "__main__":
    # change current working directory to access correct files if necessary
    if str(os.getcwd()).endswith("Code/entangled_qnn_training-main"):
        os.chdir("../../")
    
    # prepare data for convergence plots
    #extract_all_data_from_json_files()
    #fill_mean_fun_values()
    # make all convergence plots
    #make_all_convergence_plots()

    # compute all relevant convergence plot info (like achieved function values, delta, STD, etc.)
    #compute_convergence_plot_information()

    # makes boxplots for different bounds values (preliminary tests)
    make_bounds_boxplots()

    # make all category boxplots for optimizers (grad-free, evolution based, grad-based (without SGD))
    # makeCategoryBoxplots('s_rank')
    # makeCategoryBoxplots('data_type')
    # makeCategoryBoxplots('num_data_points')
