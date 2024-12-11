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

'''
    Main Experiment: Results are saved in qnn-experiments/experimental_results/results_{date}, 
    where date is the start date of the experiment in YYYY-MM-DD format
'''

no_of_runs = 10
num_layers = 1
num_qubits = 2
dimensions = 6
max_iters = [100,500,1000]
tols = [1e-5, 1e-10]

# Define default parameter bounds for optimization, covering the range [0, 2π] across all dimensions.
default_bounds = list(zip(np.zeros(6), np.ones(6)*2*np.pi))
learning_rates = [0.01, 0.001, 0.0001]

def single_optimizer_experiment(conf_id, run_id, data_type, num_data_points, s_rank, unitary, databatches, opt_list=None):
    '''
        Run all optimizer experiments for a single config & databatch combination.

        Arguments:
            conf_id (int): id of training data configuration (between 0 and 319)
            run_id (int): id of experiment run (between 0 and 9)
            data_type (String): datatype of training data
            num_data_points (int): number of training data points
            s_rank (int): Schmidt rank of training data
            unitary (tensor): unitary for ansatz
            databatches (list of tensors): batch of training data points
            opt_list (list of functions, optional): list of functions, as defined in project_qnn_experiments_optimizers.py

        Returns:
            a dict containing all specifications of optimizers & results
    '''
    start = time.time()
    result_dict = {}

    for i in range(len(databatches)):
        data_points = databatches[i]
        databatch_key = f"databatch_{i}"
        result_dict[databatch_key] = {}
        
        # Format data points as a single-line string to store in the results dictionary       
        data_points_string = (
            np.array2string(data_points.numpy(), separator=",")
            .replace("\n", "")
            .replace(" ", "")
        )
        result_dict[databatch_key]["databatch"] = data_points_string
        
        # Initialize parameter values randomly in the range [0, 2π] for optimization        
        initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions)
        initial_param_values_string = (
            np.array2string(initial_param_values, separator=",")
            .replace("\n", "")
            .replace(" ", "")
        )
        result_dict[databatch_key]["initial param values"] = initial_param_values_string
        
        #Define quantum neural network structure
        qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
        
        expected_output = torch.matmul(unitary, data_points)
        y_true = expected_output.conj()
        
        #Define the objective function for optimizers based on QNN cost function
        def objective(x):
            qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape)
            cost = cost_func(data_points, y_true, qnn, device="cpu") 
            return cost.item()
        
         # Define the objective function specifically for Particle Swarm Optimization
         # objective function must return one cost value for each particle in swarm
        def objective_for_pso(x):
            '''
            Adapted for Particle Swarm optimization.

            Arguments:
                x (array):  is of size num_particles x dimensions
            Returns:
                array of length num_particles (cost-value for each particle)
            '''
            n_particles = x.shape[0]
            cost_values = []
            for i in range(n_particles):
                qnn.params = torch.tensor(x[i], dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape)
                cost = cost_func(data_points, y_true, qnn, device="cpu") 
                cost_values.append(cost.item())
            return cost_values

        # run optimizer experiments
        sgd_optimizers = [sgd, rmsprop, adam]
        if opt_list==None:
            optimizers = [nelder_mead_experiment, bfgs_experiment, cobyla_experiment, powell_experiment, slsqp_experiment, sgd_experiment, dual_annealing_experiment, particle_swarm_experiment, genetic_algorithm_experiment, diff_evolution_experiment]
        else:
            optimizers = opt_list
        for opt in optimizers:
            if opt == sgd_experiment:
                for variant in sgd_optimizers:
                    result = sgd_experiment(objective,initial_param_values,variant)
                    opt_name = variant.__name__
                    result_dict[databatch_key][opt_name] = result
            elif opt == particle_swarm_experiment:
                result = particle_swarm_experiment(objective_for_pso)
                opt_name = opt.__name__.removesuffix('_experiment')
                result_dict[databatch_key][opt_name] = result
            else:
                result = opt(objective,initial_param_values)
                opt_name = opt.__name__.removesuffix('_experiment')
                result_dict[databatch_key][opt_name] = result
    duration = np.round((time.time()-start),2)
    print(f"config {conf_id}, run {run_id}: {duration/60}min")
    result_dict["duration (s)"] = duration
    return run_id, result_dict


def run_all_optimizer_experiments(directory, opt_list=None):
    '''
        Read all configurations of qnn and databatches from configurations_16_6_4_10_13_3_14.txt and run optimizer experiments 10 times
        for every configuration & databatch combination for each optimizer in opt_list. 
        If opt_list is None, experiments for all optimizers will be run.

        Creates json file for every configuration that saves all specifications for configuration and optimizer results.
        File is saved as "conf_{conf_id}_run_{run_id}_opt.json" in specified directory.

        Arguments:
            directory (string): where json files are to be saved
            opt_list (list of functions, optional): list of functions, as defined in project_qnn_experiments_optimizers.py
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
                    # write results to json file
                    os.makedirs(directory, exist_ok=True)
                    file = open(f"{directory}/conf_{conf_id}_run_{run_id}_opt.json", mode="w")
                    json.dump(dict, file, indent=4)

            databatches = []
            unitary = []
            n += 1

        else:
            var, val = line.split("=")
            if(var == "conf_id"): conf_id = int(val) #config ID: between 0 and 319
            elif(var == "data_type"): data_type = val # data type: random, orthogonal, non_lin_ind, var_s_rank
            elif(var == "num_data_points"): num_data_points = int(val)  # number of data points: 1, 2, 3, 4
            elif(var == "s_rank"): s_rank = int(val) # Schmidt-Rank: 1, 2, 3, 4
            elif(var == "unitary"): 
                val,_ = re.subn('\[|\]|\\n', '', val) 
                unitary = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4)) # unitary: 4x4 tensor
            elif(var.startswith("data_batch_")): 
                val,_ = re.subn('\[|\]|\\n', '', val)
                databatches.append(torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4,4))) # data_points: 1x4x4 tensor

if __name__ == "__main__":
    # change current working directory to access correct files if necessary
    if str(os.getcwd()).endswith("Code/entangled_qnn_training-main"):
        os.chdir("../../")

    start = time.time()
    print(f"start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    # run all experiments for all optimizers in opt_list
    opt_list = [nelder_mead_experiment, bfgs_experiment, cobyla_experiment, powell_experiment, slsqp_experiment, sgd_experiment, dual_annealing_experiment, genetic_algorithm_experiment, particle_swarm_experiment, diff_evolution_experiment]
    date = datetime.today().strftime('%Y-%m-%d')
    directory = f"qnn-experiments/experimental_results/results_{date}"
    run_all_optimizer_experiments(directory, opt_list=opt_list)
    print(f"total runtime (with callback): {np.round((time.time()-start)/60,2)}min") 