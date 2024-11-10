import json
import multiprocessing
import time
from datetime import datetime
from qnns.cuda_qnn import CudaPennylane

from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from victor_thesis_experiments_main import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from project_qnn_sgd_for_scipy import *
import os
from scipy.optimize import minimize, dual_annealing
import pyswarms as ps

import re


from project_qnn_experiments_optimizers import *
from project_qnn_experiments_main import *

#no_of_runs = 1
no_of_runs = 10


num_layers = 1
num_qubits = 2
dimensions = 6
max_iters = [100,500,1000]
#max_iters = [1000]
#tols = [1e-5]
tols = [1e-5, 1e-10]

#tols = [1e-10, 1e-15] schlechte ergebnisse, 1e-5 viel besser
#tols = [1e-2, 1e-5]
#bounds = [(0,2*np.pi)*dimensions]
default_bounds = list(zip(np.zeros(6), np.ones(6)*2*np.pi))
#bounds = list(zip(np.ones(6)*(-2)*np.pi, np.ones(6)*2*np.pi))
learning_rates = [0.01, 0.001, 0.0001]
#learning_rates = [0.0001]

opt_string_dict = {genetic_algorithm_experiment : "GA", diff_evolution_experiment : "DE", particle_swarm_experiment : "PSO",
                   nelder_mead_experiment: "NM", dual_annealing_experiment: "DA", sgd_experiment: "SGD", bfgs_experiment: "BFGS"}
#opt_string_dict = {genetic_algorithm_experiment : "GA", particle_swarm_experiment : "PSO"}


def run_opt_experiments_for_every_fifth_config(opt_list=None):
    '''
    Read all configurations of qnn and databatches from configurations_16_6_4_10_13_3_14.txt and run optimizer experiments
    for every fifth configuration & databatch combination, i.e. one config per combination of datatype, number of datapoints and s-rank 
    (in total 64 configs, instead of 320)
    Creates json file for every configuration that saves all specifications for configuration and optimizer results.
    File is saved as "experimental_results/results/optimizer_results/hyperparameter_tests/conf_[conf_id]_hyperparameter_tests_[opt_list].json"
    '''
    filename = "Code/entangled_qnn_training-main/experimental_results/configs/configurations_16_6_4_10_13_3_14.txt"
    file = open(filename, 'r')
    Lines = file.readlines()
    n = 0
    conf_id = 0
    databatch_id = 0
    data_type = ""
    num_data_points = 0
    s_rank = 0
    unitary = []
    databatches = []
    result_dict = {}

    for line in Lines:
        if(line.strip() == "---"): # config has been fully read and it is the right conf_id, run optimizer experiments for each data_point-tensor (5)
            if(conf_id % 5 == 0):
                # setup dictionary for dumping info into json file later
                date = datetime.now()
                result_dict_template = {"date": date.strftime("%Y/%m/%d/, %H:%M:%S"), "conf_id":conf_id, "data_type":data_type, "num_data_points":num_data_points, "s_rank":s_rank}
                unitary_string = (
                    np.array2string(unitary.numpy(), separator=",")
                    .replace("\n", "")
                    .replace(" ", "")
                )
                result_dict_template["unitary"] = unitary_string
                
                n = 0
                opt_list_string = ""
                for opt in opt_list:
                    opt_list_string = opt_list_string+"_"+opt_string_dict[opt]

                run_id,result_dict = single_optimizer_experiment(conf_id,0,data_type,num_data_points,s_rank,unitary,databatches,opt_list)
                # create complete result dictionary (begins with result_dict_template)
                dict = result_dict_template
                dict.update(result_dict)
                #write results to json file
                os.makedirs("experimental_results/results/optimizer_results/hyperparameter_tests", exist_ok=True)
                file = open(f"experimental_results/results/optimizer_results/hyperparameter_tests/conf_{conf_id}_hyperparameter_tests_{opt_list_string}.json", mode="w")
                json.dump(dict, file, indent=4)
            databatches = []
            unitary = []
            n += 1

        else:
            var, val = line.split("=")
            if(var == "conf_id"): conf_id = int(val) 
            elif(var == "data_type"): data_type = val # random, orthogonal, non_lin_ind, var_s_rank
            elif(var == "num_data_points"): num_data_points = int(val) 
            elif(var == "s_rank"): s_rank = int(val) # Schmidt-Rank
            elif(var == "unitary"): 
                val,_ = re.subn('\[|\]|\\n', '', val) 
                unitary = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4))#unitary: 4x4 tensor
            elif(var.startswith("data_batch_")): 
                val,_ = re.subn('\[|\]|\\n', '', val)
                #print(torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4)))
                databatches.append(torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4,4))) #data_points: 1x4x4 tensor



if __name__ == "__main__":
    #single_optimizer_experiment(1, "random",1,1,[],[])
    #start = time.time()
    #run_all_optimizer_experiments()
    #print(f"total runtime: {np.round((time.time()-start)/60,2)}min") 
    # total runtime: 17.59min, max_iter: 10000, optimizers = ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP']
    # total runtime: ca 40 min, max_iter = 1000, optimizers = ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP', sgd, adam, rmsprop]
    
    start = time.time()
    print(f"start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    run_opt_experiments_for_every_fifth_config(opt_list=[diff_evolution_experiment,particle_swarm_experiment,genetic_algorithm_experiment])
    print(f"total runtime (with callback): {np.round((time.time()-start)/60,2)}min") 