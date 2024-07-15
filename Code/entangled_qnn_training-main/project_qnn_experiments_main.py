import csv
import gc
import json
import time
from datetime import datetime
from qnns.cuda_qnn import CudaPennylane

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
from scipy.optimize import minimize
import re

num_layers = 1
num_qubits = 2
dimensions = 6

def single_optimizer_experiment(conf_id, data_type, num_data_points, s_rank, unitary, data_points):
    # specifications of qnn

    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()
    
    # objective function based on cost function of qnn 
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return torch.tensor(cost.item())
    optimizers = ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP', sgd, adam, rmsprop]
    #optimizers = ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP']
    results = {}
    # verschiedene inital_param_values ausprobieren und avg bilden? 
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) # [0,2pi] siehe victor_thesis_landscapes.py, bei allen optimierern gleich
    initial_param_values_tensor = torch.tensor(initial_param_values)
    for opt in optimizers:
        if(opt in ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP']):
            start = time.time()
            res = minimize(objective, initial_param_values, method=opt, 
                        options={"maxiter": 1000, "ftol":1e-10, "xtol":1e-10})
            duration = time.time() - start
        else: # funktioniert nicht! Probleme wegen jac (jacobian) & tensor
            start = time.time()
            #print(type(initial_param_values_tensor))
            res = minimize(objective, initial_param_values_tensor, method=opt,
                        options={"maxiter": 1000, "ftol":1e-10, "xtol":1e-10})
            duration = time.time() - start
        results[opt] = {'result': res, 'duration': duration}
        print(f"Optimizer: {opt}")
        print(res)
        print(f"Duration: {duration}s\n")

    #print("Results:", results)
    print("config", conf_id)

"""     with open('experimental_results/results/optimization_results.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["Optimizer", "Result", "Duration"])

        for opt, result in results.items():
            writer.writerow([opt, result['result'], result['duration']]) """

def run_single_optimizer_experiment_batch(conf_id, data_type, num_data_points, s_rank, data_batch, unitary):
    # TODO: für jede Reihe an Datenpunkten ein Experiment für alle Optimierer
    for i in range(len(data_batch)):
        data_points = data_batch[i]


def run_all_optimizer_experiments():
    filename = "Code/entangled_qnn_training-main/experimental_results/configs/configurations_16_6_4_10_13_3_14.txt"
    file = open(filename, 'r')
    Lines = file.readlines()
    n = 0
    conf_id = 0
    data_type = ""
    num_data_points = 0
    s_rank = 0
    unitary = []
    databatches = []

    for line in Lines:
        if(line.strip() == "---"): # config has been fully read, run optimizer experiments for each data_point-tensor (5)
            n = 0
            #databatch = databatches[n]
            for data_points in databatches:   
                single_optimizer_experiment(conf_id, data_type, num_data_points, s_rank, unitary, data_points)
                #print(conf_id, data_type, num_data_points, s_rank)
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



if __name__ == "__main__":
    #single_optimizer_experiment(1, "random",1,1,[],[])
    start = time.time()
    run_all_optimizer_experiments()
    print(f"total runtime: {np.round((time.time()-start)/60,2)}min") 
    # total runtime: 17.59min, max_iter: 10000, optimizers = ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP']