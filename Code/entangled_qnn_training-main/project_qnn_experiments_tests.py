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
    This file is only for testing purposes, 
    i.e. if testing one optimizer for a single configuration of training data and a single data batch is required.
'''


def test_several_optimizers():
    '''
        Test function for conf_id 0 and data_batch_0 for several optimizers
    '''
    # setup qnn configuration
    conf_id = 0
    # Choose from: "random", "orthogonal", "non_lin_ind", "var_s_rank"
    data_type = "random" 
    num_data_points = 1
    # Schmidt-Rank (Choose from: 1-4)
    s_rank = 1
    var = "data_batch_0"
    databatch_id = int(var.strip("datbch_"))

    # setup dictionary for dumping info into json file later
    result_dict = {"conf_id":conf_id, "data_type":data_type, "num_data_points":num_data_points, "s_rank":s_rank}
    val = "[[-0.45070455-0.02711853j,0.78437395-0.06613086j,0.06358678+0.19963393j,-0.07343613+0.35668523j],[-0.01890143-0.03813363j,0.32408202+0.25557629j,0.05872864-0.68979805j,0.55466693-0.20227297j],[-0.11215405+0.64023111j,-0.13344055+0.29565494j,-0.49012687-0.19046288j,-0.04241254+0.44046348j],[0.55771659+0.24656916j,0.31851997-0.05798805j,0.28761525-0.34294258j,-0.56718418+0.03616933j]]"
    result_dict["unitary"] = val#.strip("\"")
    val,_ = re.subn('\[|\]|\\n', '', val)
    unitary = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4))#unitary: 4x4 tensor
    val = '[[[0.09314128-0.12946863j,0.39382838-0.19799267j,0.05133879+0.12112185j,0.08106995-0.04021906j],[0.07622026-0.09754417j,0.31152873-0.14143589j,0.03608905+0.09551662j,0.06411194-0.02869752j],[0.11804856-0.19626647j,0.54031288-0.32976236j,0.08774511+0.16729872j,0.1112873-0.06711204j],[-0.01827577+0.10086995j,-0.17383409+0.2237231j,-0.06326177-0.05610261j,-0.03593256+0.04574145j]]]'
    result_dict[var] = {}
    result_dict[var]["databatch"] = val#.strip("\"")
    val,_ = re.subn('\[|\]|\\n', '', val) 
    data_points = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4,4)) #data_points: 1x4x4 tensor


    # Initialize the qnn
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()
    
    # objective function based on cost function of qnn 
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return cost.item()
    results = {}
     # Test various initial parameter values and potentially calculate the average performance
     # Initial parameter values within the range [0, 2π] for all optimizers (as in victor_thesis_landscapes.py)
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) 
    initial_param_values_tensor = torch.tensor(initial_param_values)

    # all results of all optimizer experiments for this config
    results = {}

    # run optimizer experiments
    sgd_optimizers = [sgd, rmsprop, adam]
    optimizers = [nelder_mead_experiment, bfgs_experiment, cobyla_experiment, powell_experiment, slsqp_experiment, sgd_experiment]
    #optimizers = [cobyla_experiment, sgd_experiment]

    #with ProcessPoolExecutor(cpu_count()) as exe:
    for opt in optimizers:
        if opt == sgd_experiment:
            for variant in sgd_optimizers:
                #future = exe.submit(sgd_experiment, objective, initial_param_values_tensor, variant)
                result = sgd_experiment(objective,initial_param_values,variant)
                opt_name = variant.__name__
                #result_dict[var][opt_name] = future.result()
                result_dict[var][opt_name] = result
        else:
            #future = exe.submit(opt, objective, initial_param_values)
            result = opt(objective,initial_param_values)
            opt_name = opt.__name__.removesuffix('_experiment')
            #result_dict[var][opt_name] = future.result()
            result_dict[var][opt_name] = result

    # Save results in json file "conf_[conf_id]_opt.json"
    os.makedirs("experimental_results/results/optimizer_results", exist_ok=True)
    file = open(f"experimental_results/results/optimizer_results/conf_{conf_id}_test.json", mode="w")
    json.dump(result_dict, file)

def test_single_optimizer(callback: bool):
    '''
    Test function for conf_id 0 and data_batch_0 for one optimizer.
    '''

    # Setup qnn configuration
    conf_id = 0
    # Choose from: "random", "orthogonal", "non_lin_ind", "var_s_rank"
    data_type = "random" 
    num_data_points = 1
    # Schmidt-Rank (Choose from: 1-4)
    s_rank = 1
    var = "data_batch_0"
    databatch_id = int(var.strip("datbch_"))

    # setup dictionary for dumping info into json file later
    result_dict = {"conf_id":conf_id, "data_type":data_type, "num_data_points":num_data_points, "s_rank":s_rank}
    val = "[[-0.45070455-0.02711853j,0.78437395-0.06613086j,0.06358678+0.19963393j,-0.07343613+0.35668523j],[-0.01890143-0.03813363j,0.32408202+0.25557629j,0.05872864-0.68979805j,0.55466693-0.20227297j],[-0.11215405+0.64023111j,-0.13344055+0.29565494j,-0.49012687-0.19046288j,-0.04241254+0.44046348j],[0.55771659+0.24656916j,0.31851997-0.05798805j,0.28761525-0.34294258j,-0.56718418+0.03616933j]]"
    result_dict["unitary"] = val
    val,_ = re.subn('\[|\]|\\n', '', val)
    unitary = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4)) #unitary: 4x4 tensor
    val = '[[[0.09314128-0.12946863j,0.39382838-0.19799267j,0.05133879+0.12112185j,0.08106995-0.04021906j],[0.07622026-0.09754417j,0.31152873-0.14143589j,0.03608905+0.09551662j,0.06411194-0.02869752j],[0.11804856-0.19626647j,0.54031288-0.32976236j,0.08774511+0.16729872j,0.1112873-0.06711204j],[-0.01827577+0.10086995j,-0.17383409+0.2237231j,-0.06326177-0.05610261j,-0.03593256+0.04574145j]]]'
    result_dict[var] = {}
    result_dict[var]["databatch"] = val#.strip("\"")
    val,_ = re.subn('\[|\]|\\n', '', val) 
    data_points = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4,4)) #data_points: 1x4x4 tensor


    # setup qnn
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()
    
    # objective function based on cost function of qnn 
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) 
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return cost.item()
    results = {}
    
    # Generate initial parameter values for the qnn from the range [0, 2*pi]
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions)
    
    # Run the optimization using the Nelder-Mead optimizer
    # include a callback function to save intermediate results during the optimization process 
    if callback==True:
        res = minimize(objective, initial_param_values, method='Nelder-Mead', callback=saveIntermResult,
                        options={"maxiter": 1000, "eps":1e-5})
        print(fun_all)
        fun_all.clear()
        global nit
        nit = 0
    else:
        res = minimize(objective, initial_param_values, method='Nelder-Mead',
                        options={"maxiter": 1000, "eps":1e-5})


if __name__ == "__main__":
    start = time.time()
    print(f"start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    test_single_optimizer(False)
    print(f"total runtime (with callback): {np.round((time.time()-start)/60,2)}min") 