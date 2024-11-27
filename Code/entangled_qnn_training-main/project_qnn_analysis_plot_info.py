import csv
import gc
import json
import time
from datetime import datetime
import matplotlib.patches
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

conf_ids_to_skip = [190, 191, 192, 193, 194, 210, 211, 212, 213, 214, 230, 231, 232, 233, 234]
combinations_to_skip = [["non_lin_ind","2","3"],["non_lin_ind","3","3"],["non_lin_ind","4","3"]] # Format [data_type, num_data_points, s_rank]

opt_titles = {'nelder_mead': 'Nelder-Mead', 'powell':'Powell', 'sgd':'SGD', 
              'adam':'Adam', 'rmsprop':'RMSprop', 'bfgs':'BFGS','slsqp':'SLSQP',
              'dual_annealing':'Dual Annealing','cobyla':'COBYLA',
              'genetic_algorithm':'Genetic Algorithm', 'particle_swarm': 'Particle Swarm Optimization',
              'diff_evolution':'Differential Evolution'}

datatype_list = ['random', 'orthogonal', 'non_lin_ind', 'var_s_rank']
num_data_points_list = ['1', '2', '3', '4']
s_rank_list = ['1', '2', '3', '4']
maxiter_list = [1000]


def check_deltas_for(var_param):
    '''
        Analyses the difference between values for var_param in optimizer results.

        Requirement:
            Convergence plots and all corresponding csv files in ../plot_info/ exist
    '''
    # check that only one parameter (data_type, num_data_points, s_rank) is None:
    param_value_list = {'data_type': datatype_list, 'num_data_points': [1,2,3,4], 's_rank': [1,2,3,4]}
    
    # determine the non-variable parameters
    path_prefix = 'qnn-experiments/plots/convergence_plots/maxiter/1000/plot_info/'
    if var_param == "data_type":
        directory = path_prefix+f"s_rank/"
        param0 = "s_rank"
        param1 = "num_data_points"
    elif var_param == "s_rank":
        directory = path_prefix+f"datatype/"
        param0 = "data_type"
        param1 = "num_data_points"
    elif var_param == "num_data_points":
        directory = path_prefix+f"datatype/"
        param0 = "data_type"
        param1 = "s_rank"
    else:
        raise Exception(f"{var_param} is not a valid argument.")


    param_list0 = param_value_list[param0]
    param_list1 = param_value_list[param1]

    filename = "delta_mean_fun_values"
    df_list_1 = []
    df_list_2 = []
    df_list_3 = []
    print(f"Variable {var_param}...")
    for value in param_list0:
        dir = directory + f"{value}/{param1}"
        for filename in os.listdir(dir):
            if filename.startswith("rel_delta_mean_fun_values"):
                file_path = os.path.join(dir, filename)
                d = pd.read_csv(file_path)
                if(d.shape[0]>1):
                    df_list_1.append(d.iloc[[1]])
                    df_list_2.append(d.iloc[[2]])
                if(d.shape[0] == 4):
                    df_list_3.append(d.iloc[[3]])
    df1 = pd.concat(df_list_1).set_index(['data_type', 'num_data_points', 's_rank'])
    df2 = pd.concat(df_list_2).set_index(['data_type', 'num_data_points', 's_rank'])
    df3 = pd.concat(df_list_3).set_index(['data_type', 'num_data_points', 's_rank'])
    df_list = [df1, df2, df3]
    text_list = [f"Relative Difference between {var_param} 1 and 2", f"Relative Difference between {var_param} 2 and 3", f"Relative Difference between {var_param} 3 and 4"]
    for i in range(3):
        df_complete = df_list[i]
        print(text_list[i])
        # for all relative deltas (independent of datatype and num data points)
        print("For all values: ")
        print("Min", df_complete[opt_titles.keys()].min(axis=None))
        print("Max", df_complete[opt_titles.keys()].max(axis=None))
        print("Mean", df_complete[opt_titles.keys()].mean(axis=None))
        print("STD", df_complete[opt_titles.keys()].stack().std())
        # dependent on datatype
        print(f"\nFor variable {param0}:")
        for value in param_list0:
            print("---"+str(value)+"---")
            df = df_complete[df_complete.index.get_level_values(param0) == value]
            print("Min", df[opt_titles.keys()].min(axis=None))
            print("Max", df[opt_titles.keys()].max(axis=None))
            print("Mean", df[opt_titles.keys()].mean(axis=None))
            print("STD", df[opt_titles.keys()].stack().std())
        # dependent on num data points
        print(f"\nFor variable {param1}:")
        for value in param_list1:
            print("---"+str(value)+"---")
            df = df_complete[df_complete.index.get_level_values(param1) == value]
            print("Min", df[opt_titles.keys()].min(axis=None))
            print("Max", df[opt_titles.keys()].max(axis=None))
            print("Mean", df[opt_titles.keys()].mean(axis=None))
            print("STD", df[opt_titles.keys()].stack().std())
        print("============================")

def get_mean_and_delta_table_info(var_param):
    '''
        Computes mean optimizer results and relative differences for variable var_param for each optimizer.
        Needed for tables in chapter 5.2 in the paper.

        Requirement:
            Convergence plots and all corresponding csv files in ../plot_info/ exist
    '''
    # check that only one parameter (data_type, num_data_points, s_rank) is None:
    param_value_list = {'data_type': datatype_list, 'num_data_points': [1,2,3,4], 's_rank': [1,2,3,4]}
    
    # determine the non-variable parameters
    path_prefix = 'qnn-experiments/plots/convergence_plots/maxiter/1000/plot_info/'
    if var_param == "data_type":
        directory = path_prefix+f"s_rank/"
        param0 = "s_rank"
        param1 = "num_data_points"
    elif var_param == "s_rank":
        directory = path_prefix+f"datatype/"
        param0 = "data_type"
        param1 = "num_data_points"
    elif var_param == "num_data_points":
        directory = path_prefix+f"datatype/"
        param0 = "data_type"
        param1 = "s_rank"
    else:
        raise Exception(f"{var_param} is not a valid argument.")


    param_list0 = param_value_list[param0]
    param_list1 = param_value_list[param1]

    filename = "delta_mean_fun_values"
    delta_list_1, delta_list_2, delta_list_3 = [], [], []
    mean_list_0, mean_list_1, mean_list_2, mean_list_3 = [], [], [], []
    print(f"Variable {var_param}...")
    for value in param_list0:
        dir = directory + f"{value}/{param1}"
        for filename in os.listdir(dir):
            if filename.startswith("rel_delta_mean_fun_values"):
                file_path = os.path.join(dir, filename)
                d = pd.read_csv(file_path)
                if(d.shape[0]>1):
                    delta_list_1.append(d.iloc[[1]])
                    delta_list_2.append(d.iloc[[2]])
                if(d.shape[0] == 4):
                    delta_list_3.append(d.iloc[[3]])
            if filename.startswith("mean_fun_values"):
                file_path = os.path.join(dir, filename)
                d = pd.read_csv(file_path)
                mean_list_0.append(d.iloc[[0]])
                if(d.shape[0]>1):
                    mean_list_1.append(d.iloc[[1]])
                    mean_list_2.append(d.iloc[[2]])
                if(d.shape[0] == 4):
                    mean_list_3.append(d.iloc[[3]])
    delta_df1 = pd.concat(delta_list_1).set_index(['data_type', 'num_data_points', 's_rank'])
    delta_df2 = pd.concat(delta_list_2).set_index(['data_type', 'num_data_points', 's_rank'])
    delta_df3 = pd.concat(delta_list_3).set_index(['data_type', 'num_data_points', 's_rank'])
    mean_df0 = pd.concat(mean_list_0).set_index(['data_type', 'num_data_points', 's_rank'])
    mean_df1 = pd.concat(mean_list_1).set_index(['data_type', 'num_data_points', 's_rank'])
    mean_df2 = pd.concat(mean_list_2).set_index(['data_type', 'num_data_points', 's_rank'])
    mean_df3 = pd.concat(mean_list_3).set_index(['data_type', 'num_data_points', 's_rank'])
    delta_list = [delta_df1, delta_df2, delta_df3]
    mean_list = [mean_df0, mean_df1, mean_df2, mean_df3]
    text_list = [f"Relative Difference between {var_param} 1 and 2", f"Relative Difference between {var_param} 2 and 3", f"Relative Difference between {var_param} 3 and 4"]
    
    print("First Value:")
    mean_complete = mean_list[0]
    # for all relative deltas (independent of datatype and num data points)
    print("For all values: ")
    print("Mean", mean_complete[opt_titles.keys()].mean(axis=None), "STD", mean_complete[opt_titles.keys()].stack().std())
    print("Mean for each optimizer:")
    print(mean_complete.mean())
    for i in range(3):
        delta_complete = delta_list[i]
        mean_complete = mean_list[i+1]
        print(text_list[i])
        # for all relative deltas (independent of datatype and num data points)
        print("For all values: ")
        print("Mean", mean_complete[opt_titles.keys()].mean(axis=None), "delta", delta_complete[opt_titles.keys()].mean(axis=None), "STD", mean_complete[opt_titles.keys()].stack().std())
        print("Mean for each optimizer:")
        print(mean_complete.mean())
        print("Delta for each optimizer:")
        print(delta_complete.mean())

def get_datatype_distribution():
    '''
        Calculates the percentage of each datatype for each "place" in the convergence plot. 
        I.e. non_lin_ind was the best datatype (place 1) 50% of the time, orthogonal 20% of the time, etc.
        
        Requirement:
            Convergence plots and all corresponding csv files in ../plot_info/ exist
    '''
    path = 'qnn-experiments/plots/convergence_plots/maxiter/1000/plot_info/s_rank/'

    type_list = {1: [], 2: [], 3: [], 4: []}
    print(f"Variable datatype...")
    # exclude S-rank = 3, since no values for non_lin_ind in that case
    for value in [1,2,4]:
        dir = path + f"{value}/num_data_points"
        for filename in os.listdir(dir):
            # exclude 1 data point from analysis, since datatype has no effect on 1 datapoint.
            if (filename.startswith("order_paramvalue_per_opt_None4") or filename.startswith("order_paramvalue_per_opt_None3") or filename.startswith("order_paramvalue_per_opt_None2")):
                file_path = os.path.join(dir, filename)
                d = pd.read_csv(file_path, index_col=0)
                for i in range(d.shape[0]):
                    type_list[i+1].append(d.iloc[[i]])
    type_df_list = [pd.concat(type_list[i+1]) for i in range(4)]

    for i in range(4):
        print(f"Place {i+1}:")
        df = type_df_list[i].stack().value_counts(normalize=True) * 100
        print(df)
        print("=============")

def get_mean_std_variable_datatype(num_data_points=1):
    '''
        Calculates the average optimizer result and std for variable datatype but fixed number of data points (default: 1)

        Requirement:
            Convergence plots and all corresponding csv files in ../plot_info/ exist
    '''
    path = 'qnn-experiments/plots/convergence_plots/maxiter/1000/plot_info/s_rank/'
    mean_list = {1: [], 2: [], 3: [], 4: []}
    for value in [1,2,3,4]:
        dir = path + f"{value}/num_data_points"
        for filename in os.listdir(dir):
            if filename.startswith(f"mean_fun_values_None{num_data_points}"):
                file_path = os.path.join(dir, filename)
                d = pd.read_csv(file_path).set_index(['data_type', 'num_data_points', 's_rank'])
                mean_list[value].append(d)
    mean_df_list = [pd.concat(mean_list[value]) for value in [1,2,3,4]]

    print(f"variable datatype, number of datapoints: {num_data_points}")
    std = 0
    n = 0
    for value in range(4):
        print(f"Schmidt-Rank: {value+1}")
        df = mean_df_list[value].describe()
        std += df.loc["std"].sum(axis=0)
        n += 12 # mean_df_list[value] has 12 columns (12 optimizers)
        print(df)
        print("==========")
    print("mean STD: ", std/n)
  #  

def get_delta_non_lin_ind_to_others():
    '''
        Computes relative distance between achieved function value for non_lin_ind and other datatypes for Schmidt-Rank 1 and 2 and more than one data point.
        
        Requirement:
            Convergence plots and all corresponding csv files in ../plot_info/ exist
    '''
    path = 'qnn-experiments/plots/convergence_plots/maxiter/1000/plot_info/s_rank/'

    type_list = {1: [], 2: [], 3: [], 4: []}
    delta_list = []
    print(f"Variable datatype, Schmidt-Rank 1 and 2, number of data points > 1")
    for value in [1,2]:
        dir = path + f"{value}/num_data_points"
        for filename in os.listdir(dir):
            # exclude 1 data point from analysis, since datatype has no effect on 1 datapoint.
            if (filename.startswith("mean_fun_values_None4") or filename.startswith("mean_fun_values_None3") or filename.startswith("mean_fun_values_None2")):
                file_path = os.path.join(dir, filename)
                d = pd.read_csv(file_path).set_index("data_type")
                #for each column (optimizer) compute difference of result for other datatypes to non_lin_ind
                for datatype in ["random", "var_s_rank", "orthogonal"]:
                    df = (d.loc[datatype]-d.loc["non_lin_ind"])/d.loc["non_lin_ind"] # compute relative difference
                    delta_list.append(df[opt_titles.keys()])
    delta_df = pd.concat(delta_list)
    #mean_df_list = [pd.concat(mean_list[datatype]).set_index(['data_type', 'num_data_points', 's_rank']) for datatype in datatype_list]
    mean_delta = delta_df.describe()
    print(mean_delta)
    


if __name__ == "__main__":
    # change current working directory to access correct files
    os.chdir("../../")

    # print("Further Convergence Plot info")
    # print("================================= Check Deltas =================================")
    # check_deltas_for("s_rank")
    # check_deltas_for("num_data_points")

    # print("================================= Tables =================================")
    # get_mean_and_delta_table_info("s_rank")
    # get_mean_and_delta_table_info("num_data_points")

    # print("================================= Datatype table =================================")
    # get_datatype_distribution()

    # print("================================= STD for datatype (ndp=1) =================================")
    # get_mean_std_variable_datatype()

    print("================================= Delta non_lin_ind to others =================================")
    get_delta_non_lin_ind_to_others()
    