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
from sgd_for_scipy import *
import os
from scipy.optimize import minimize, dual_annealing
import pyswarms as ps

import re


from project_qnn_experiments_optimizers import *

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

def single_optimizer_experiment(conf_id, run_id, data_type, num_data_points, s_rank, unitary, databatches, opt_list=None):
    '''
    Run all optimizer experiments for a single config & databatch combination

    Return:
        dict containing all specifications of optimizers & results
    '''
    start = time.time()
    result_dict = {}

    for i in range(len(databatches)):
        data_points = databatches[i]
        databatch_key = f"databatch_{i}"
        result_dict[databatch_key] = {}
        data_points_string = (
            np.array2string(data_points.numpy(), separator=",")
            .replace("\n", "")
            .replace(" ", "")
        )
        result_dict[databatch_key]["databatch"] = data_points_string
        initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions)
        initial_param_values_string = (
            np.array2string(initial_param_values, separator=",")
            .replace("\n", "")
            .replace(" ", "")
        )
        result_dict[databatch_key]["initial param values"] = initial_param_values_string
        
        # specifications of qnn
        qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
        
        expected_output = torch.matmul(unitary, data_points)
        y_true = expected_output.conj()
        
        # objective function based on cost function of qnn 
        def objective(x):
            qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
            cost = cost_func(data_points, y_true, qnn, device="cpu") 
            return cost.item()
        
        # objective function for Particle Swarm Optimization
        def objective_for_pso(x):
            '''
            Adapted for Particle Swarm optimization.
            x  is of size num_particles x dimensions
            returns array of length num_particles (cost-value for each particle)
            '''
            n_particles = x.shape[0]
            cost_values = []
            for i in range(n_particles):
                qnn.params = torch.tensor(x[i], dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
                cost = cost_func(data_points, y_true, qnn, device="cpu") 
                cost_values.append(cost.item())
            return cost_values

        # verschiedene inital_param_values ausprobieren und avg bilden? 
        #initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) # [0,2pi] siehe victor_thesis_landscapes.py, bei allen optimierern gleich
        #initial_param_values_tensor = torch.tensor(initial_param_values)

        # run optimizer experiments
        sgd_optimizers = [sgd, rmsprop, adam]
        #sgd_optimizers = [adam]
        if opt_list==None:
            optimizers = [nelder_mead_experiment, bfgs_experiment, cobyla_experiment, powell_experiment, slsqp_experiment, sgd_experiment, dual_annealing_experiment]
        else:
            optimizers = opt_list
        
        # TODO: ProcessPoolExecutor: funktioniert nicht, weil pickle verwendet wird und objective eine lokal definierte Funktion ist 
        # (AttributeError: Can't pickle local object 'test_experiment.<locals>.objective')
        # Multiprocessing (??) könnte funktionieren. Oder eigene Klasse??
        #with ProcessPoolExecutor(cpu_count()) as exe:
        #with multiprocessing.pool.Pool() as pool:

        for opt in optimizers:
            if opt == sgd_experiment:
                for variant in sgd_optimizers:
                    #future = exe.submit(sgd_experiment, objective, initial_param_values, variant)
                    result = sgd_experiment(objective,initial_param_values,variant)
                    #future = pool.map_async(sgd_experiment, (objective, initial_param_values, variant,))
                    opt_name = variant.__name__
                    #result_dict[opt_name] = future.get()
                    #result_dict[opt_name] = future.result()
                    result_dict[databatch_key][opt_name] = result
            elif opt == particle_swarm_experiment:
                result = particle_swarm_experiment(objective_for_pso)
                opt_name = opt.__name__.removesuffix('_experiment')
                result_dict[databatch_key][opt_name] = result
            else:
                #future = exe.submit(opt, objective, initial_param_values)
                result = opt(objective,initial_param_values)
                #future = pool.map_async(opt, (objective, initial_param_values,))
                opt_name = opt.__name__.removesuffix('_experiment')
                #result_dict[opt_name] = future.get()
                #result_dict[opt_name] = future.result()
                result_dict[databatch_key][opt_name] = result
    duration = np.round((time.time()-start),2)
    print(f"config {conf_id}, run {run_id}: {duration/60}min")
    result_dict["duration (s)"] = duration
    return run_id, result_dict


def run_all_optimizer_experiments(opt_list=None):
    '''
    Read all configurations of qnn and databatches from configurations_16_6_4_10_13_3_14.txt and run optimizer experiments
    for every configuration & databatch combination
    Creates json file for every configuration that saves all specifications for configuration and optimizer results.
    File is saved as "experimental_results/results/optimizer_results/conf_[conf_id]_opt.json"
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
        if(line.strip() == "---"): # config has been fully read, run optimizer experiments for each data_point-tensor (5)
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
            with ProcessPoolExecutor(max_workers=5) as exe:
                futures = [exe.submit(single_optimizer_experiment,conf_id, run_id, data_type, num_data_points, s_rank, unitary, databatches,opt_list) for run_id in range(no_of_runs)]            
                for future in as_completed(futures):
                    # get the result for the next completed task
                    run_id, result_dict = future.result()# blocks
                    # create complete result dictionary (begins with result_dict_template)
                    dict = result_dict_template
                    dict.update(result_dict)
                    #write results to json file
                    os.makedirs("experimental_results/results/optimizer_results/GA_PSO_DE", exist_ok=True)
                    file = open(f"experimental_results/results/optimizer_results/GA_PSO_DE/conf_{conf_id}_run_{run_id}_opt.json", mode="w")
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
    run_all_optimizer_experiments(opt_list=[genetic_algorithm_experiment, particle_swarm_experiment, diff_evolution_experiment])
    print(f"total runtime (with callback): {np.round((time.time()-start)/60,2)}min") 