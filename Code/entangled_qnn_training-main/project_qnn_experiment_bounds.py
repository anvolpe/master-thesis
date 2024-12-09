import json
import time
from datetime import datetime
from qnns.cuda_qnn import CudaPennylane

from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from victor_thesis_experiments_main import *

from project_qnn_experiments_optimizers import *
from project_qnn_sgd_for_scipy import *
import os
import re

num_layers = 1
num_qubits = 2
dimensions = 6
max_iters = [100,500,1000]
tols = [1e-5, 1e-10]
default_bounds = list(zip(np.zeros(6), np.ones(6)*2*np.pi))
learning_rates = [0.01, 0.001, 0.0001]

def single_config_experiment_bounds(conf_id, databatch_id, data_type, num_data_points, s_rank, unitary, data_points):
    '''
    Run all optimizer experiments for a single config & databatch combination

    Arguments:
        conf_id (int): id of configuration of training data, between 0 and 319
        databatch_id (int): number of databatch, between 0 and 4
        data_type (String): datatype of underlying training data  one of 'random', 'orthogonal', 'non_lin_ind', 'var_s_rank'
        num_data_points (String): number of data points, one of 1, 2, 3, 4
        s_rank (String): Schmidt rank of training data, one of 1, 2, 3, 4
        unitary (tensor): shape: 4x4
        data_points (tensor): shape 1x4x4

    Returns:
        dict containing all specifications of optimizers & results
    '''
    result_dict = {}
    data_points_string = (
        np.array2string(data_points.numpy(), separator=",")
        .replace("\n", "")
        .replace(" ", "")
    )
    result_dict["databatch"] = data_points_string
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions)
    initial_param_values_string = (
        np.array2string(initial_param_values, separator=",")
        .replace("\n", "")
        .replace(" ", "")
    )
    result_dict["initial param values"] = initial_param_values_string
    
    # specifications of qnn
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()
    
    # objective function based on cost function of qnn 
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return cost.item()

    # verschiedene inital_param_values ausprobieren und avg bilden? 
    #initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) # [0,2pi] siehe victor_thesis_landscapes.py, bei allen optimierern gleich
    #initial_param_values_tensor = torch.tensor(initial_param_values)

    # run optimizer experiments
    sgd_optimizers = [sgd, rmsprop, adam]
    optimizers = [nelder_mead_experiment, bfgs_experiment, cobyla_experiment, powell_experiment, slsqp_experiment, sgd_experiment, dual_annealing_experiment]
    bound_batches = []
    bound_batches.append(list(zip(np.zeros(6), np.ones(6)*2*np.pi)))
    bound_batches.append(list(zip(np.zeros(6), np.ones(6)*4*np.pi)))
    bound_batches.append(list(zip(np.ones(6)*(-2)*np.pi, np.ones(6)*2*np.pi)))
    bound_batches.append(list(zip(np.ones(6)*(-4)*np.pi, np.ones(6)*4*np.pi)))


    # no bounds for all optimizers (excluding dual annealing):
    # maxiter = 1000, tol = 1e-5, learning rate = 0.001
    result_dict["bounds_0"] = {"bounds": "none"}
    for opt in optimizers:
        if opt == sgd_experiment:
            for variant in sgd_optimizers:
                result = sgd_experiment(objective,initial_param_values,variant)
                opt_name = variant.__name__
                result_dict["bounds_0"][opt_name] = result
        elif opt != dual_annealing_experiment:
            result = opt(objective,initial_param_values)
            opt_name = opt.__name__.removesuffix('_experiment')
            result_dict["bounds_0"][opt_name] = result

    # for all bounds test all optimizers (except sgd, adam, rmsprop and bfgs, since no bounds can be specified)
    # maxiter = 1000, tol = 1e-5, learning rate = 0.001
    optimizers = [nelder_mead_experiment, cobyla_experiment, powell_experiment, slsqp_experiment, dual_annealing_experiment]
    i = 1
    for bounds in bound_batches:
        result_dict[f'bounds_{i}'] = {"bounds": bounds}
        for opt in optimizers:
            result = opt(objective,initial_param_values,bounds)
            opt_name = opt.__name__.removesuffix('_experiment')
            result_dict[f'bounds_{i}'][opt_name] = result
        i += 1

    return result_dict


def test_bounds():
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
    i = 1
    for line in Lines:
        if(i>100): break
        if(line.strip() == "---"): # config has been fully read, run optimizer experiments for each data_point-tensor (5)
            # setup dictionary for dumping info into json file later
            date = datetime.now()
            result_dict = {"date": date.strftime("%Y/%m/%d/, %H:%M:%S"), "conf_id":conf_id, "data_type":data_type, "num_data_points":num_data_points, "s_rank":s_rank}
            unitary_string = (
                np.array2string(unitary.numpy(), separator=",")
                .replace("\n", "")
                .replace(" ", "")
            )
            result_dict["unitary"] = unitary_string
            n = 0
            #databatch = databatches[n]
            #print(len(databatches))
            start = time.time()
            for i in range(len(databatches)): 
                data_points = databatches[i]
                dict = single_config_experiment_bounds(conf_id, i, data_type, num_data_points, s_rank, unitary, data_points)
                databatch_key = f"databatch_{i}"
                result_dict[databatch_key] = dict
                #print(conf_id, data_type, num_data_points, s_rank)
            #write results to json file
            duration = np.round((time.time()-start),2)
            print(f"config {conf_id}: {duration/60}min")
            result_dict["duration (s)"] = duration
            os.makedirs("experimental_results/results/optimizer_results/bounds", exist_ok=True)
            file = open(f"experimental_results/results/optimizer_results/bounds/conf_{conf_id}_bounds.json", mode="w")
            json.dump(result_dict, file)
            databatches = []
            unitary = []
            n += 1

        else:
            var, val = line.split("=")
            if(var == "conf_id"): conf_id = int(val) 
            elif(var == "data_type"): data_type = val # random, orthogonal, non_lin_ind, var_s_rank
            elif(var == "num_data_points"): num_data_points = int(val) 
            elif(var == "s_rank"): s_rank = int(s_rank) # Schmidt-Rank
            elif(var == "unitary"): 
                val,_ = re.subn('\[|\]|\\n', '', val) 
                unitary = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4))#unitary: 4x4 tensor
            elif(var.startswith("data_batch_")): 
                val,_ = re.subn('\[|\]|\\n', '', val)
                #print(torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4)))
                databatches.append(torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4,4))) #data_points: 1x4x4 tensor
        i += 1

if __name__ == "__main__":
    start = time.time()
    print(f"start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    test_bounds()
    print(f"total runtime (with callback): {np.round((time.time()-start)/60,2)}min") 