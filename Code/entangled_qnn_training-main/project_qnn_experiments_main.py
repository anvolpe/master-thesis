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
max_iters = [100,500,1000]
tols = [1e-3, 1e-5, 1e-10]

def nelder_mead_experiment(objective,initial_param_values):
    results = {}
    for max_iter in max_iters:
        for fatol in tols:
            for xatol in tols:
                start = time.time()
                res = minimize(objective, initial_param_values, method="Nelder-Mead", 
                        options={"maxiter": max_iter, "fatol":fatol, "xatol":xatol})
                duration = time.time() - start

    return results

def single_config_experiments(conf_id, data_type, num_data_points, s_rank, unitary, data_points):
    # prepare csv file for experiment results
    os.makedirs("experimental_results/results/optimizer_results", exist_ok=True)
    file = open(f"experimental_results/results/optimizer_results/conf_{conf_id}_opt.csv", mode="w")
    writer = csv.writer(file)
    writer.writerow(["Optimizer", "Result", "Duration"])
    # TODO: Infos zu Config speichern (alles was oben als Argument übergeben wird)
    # evtl json statt csv? oder txt?
    
    # specifications of qnn
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
        
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()

    # objective function based on cost function of qnn 
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return torch.tensor(cost.item())
    # TODO: verschiedene inital_param_values ausprobieren und avg bilden? (zuerst schauen wie lang es dauert)
    # same initial_param_values for all optimizers
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) # [0,2pi] siehe victor_thesis_landscapes.py, bei allen optimierern gleich
    initial_param_values_tensor = torch.tensor(initial_param_values) 

    # alle optimierer experiments laufen lassen
    # return ist jeweils result-dict (das auch alle Spezifikationen enthält?)
    # das in der Datei conf_[conf_id]_opt.csv speichern

        
''' VERALTET?? aber funktioniert'''
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
    max_iters = [100,500,1000]
    tols = [1e-3, 1e-5, 1e-10]
    
    for opt in optimizers:
        
        if(opt in ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP']): # TODO: Idee: pro Optimierer eigene Funktion, wo Spezifikationen auf diesen Optimierer abgestimmt sind? (da verschiedene Parameternamen)
            start = time.time()
            res = minimize(objective, initial_param_values, method=opt, 
                        options={"maxiter": 1000, "ftol":1e-10, "xtol":1e-10})
            duration = time.time() - start
        else:
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

    with open('experimental_results/results/optimization_results.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["Optimizer", "Result", "Duration"])
        # TODO: einzelne Spezifikationen in file speichern
        # evtl json statt csv?
        for opt, result in results.items():
            writer.writerow([opt, result['result'], result['duration']])

''' VERALTET?? unnötig?'''
def run_single_optimizer_experiment_batch(conf_id, data_type, num_data_points, s_rank, data_batch, unitary):
    # TODO: für jede Reihe an Datenpunkten ein Experiment für alle Optimierer
    for i in range(len(data_batch)):
        data_points = data_batch[i]

''' VERALTET?? aber funktioniert'''
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
    # total runtime: ca 40 min, max_iter = 1000, optimizers = ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP', sgd, adam, rmsprop]