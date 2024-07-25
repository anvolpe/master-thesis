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

class QNN_Experiment:
    num_layers = 1
    num_qubits = 2
    dimensions = 6
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    data_points = []
    y_true = []


    max_iters = [100,500,1000]
    tols = [1e-5, 1e-10]
    bounds = []
    learning_rates = [0.01, 0.001, 0.0001]

    def objective(self, x):
        self.qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(self.qnn.params.shape) # stimmt das???????
        cost = cost_func(self.data_points, self.y_true, self.qnn, device="cpu") 
        return torch.tensor(cost.item())

    def nelder_mead_experiment(self,initial_param_values):
        results = {}
        run_n = 0
        for max_iter in self.max_iters:
            for fatol in self.tols:
                for xatol in self.tols:
                    start = time.time()
                    res = minimize(self.objective, initial_param_values, method="Nelder-Mead", 
                            options={"maxiter": max_iter, "fatol":fatol, "xatol":xatol})
                    duration = time.time() - start
                    # fill results dict
                    # specifications of this optimizer run
                    results[run_n] = {"maxiter": max_iter, "fatol":fatol, "xatol":xatol, "duration":duration}
                    # result info
                    for attribute in res.keys():
                        results[run_n][attribute] = str(res[attribute])
                    run_n += 1
        return results

    def cobyla_experiment(objective,initial_param_values):
        results = {}
        run_n = 0
        for max_iter in self.max_iters:
            for tol in self.tols:
                for catol in self.tols:
                    start = time.time()
                    res = minimize(objective, initial_param_values, method="COBYLA", 
                            options={"maxiter": max_iter, "tol":tol, "catol":catol})
                    duration = time.time() - start
                    # fill results dict
                    # specifications of this optimizer run
                    results[run_n] = {"maxiter": max_iter, "tol":tol, "catol":catol, "duration":duration}
                    # result info
                    for attribute in res.keys():
                        results[run_n][attribute] = str(res[attribute])
                    run_n += 1
        return results

    def bfgs_experiment(objective,initial_param_values):
        results = {}
        run_n = 0
        for max_iter in self.max_iters:
            for gtol in self.tols:
                for xrtol in self.tols:
                    for eps in self.tols:
                        start = time.time()
                        res = minimize(objective, initial_param_values, method="BFGS", 
                                options={"maxiter": max_iter, "gtol":gtol, "xrtol":xrtol, "eps":eps})
                        duration = time.time() - start
                        # fill results dict
                        # specifications of this optimizer run
                        results[run_n] = {"maxiter": max_iter, "gtol":gtol, "xrtol":xrtol, "eps":eps, "duration":duration}
                        # result info
                        for attribute in res.keys():
                            results[run_n][attribute] = str(res[attribute])
                        run_n += 1
        return results

    def powell_experiment(objective,initial_param_values):
        results = {}
        run_n = 0
        for max_iter in max_iters:
            for ftol in tols:
                for xtol in tols:
                    start = time.time()
                    res = minimize(objective, initial_param_values, method="Powell", 
                            options={"maxiter": max_iter, "ftol":ftol, "xtol":xtol})
                    duration = time.time() - start
                    # fill results dict
                    # specifications of this optimizer run
                    results[run_n] = {"maxiter": max_iter, "ftol":ftol, "xtol":xtol, "duration":duration}
                    # result info
                    for attribute in res.keys():
                        results[run_n][attribute] = str(res[attribute])
                    run_n += 1
        return results

    def slsqp_experiment(objective,initial_param_values):
        results = {}
        run_n = 0
        for max_iter in max_iters:
            for ftol in tols:
                for eps in tols:
                    start = time.time()
                    res = minimize(objective, initial_param_values, method="SLSQP", 
                            options={"maxiter": max_iter, "ftol":ftol, "eps":eps})
                    duration = time.time() - start
                    # fill results dict
                    # specifications of this optimizer run
                    results[run_n] = {"maxiter": max_iter, "ftol":ftol, "eps":eps, "duration":duration}
                    # result info
                    for attribute in res.keys():
                        results[run_n][attribute] = str(res[attribute])
                    run_n += 1
        return results

    def sgd_experiment(objective,initial_param_values,opt):
        results = {}
        run_n = 0
        for max_iter in max_iters:
            for learning_rate in learning_rates:
                for eps in tols:
                    start = time.time()
                    res = minimize(objective, initial_param_values, method=opt, 
                            options={"maxiter": max_iter, "learning_rate":learning_rate, "eps":eps})
                    duration = time.time() - start
                    # fill results dict
                    # specifications of this optimizer run
                    results[run_n] = {"maxiter": max_iter, "learning_rate":learning_rate, "eps":eps, "duration":duration}
                    # result info
                    for attribute in res.keys():
                        results[run_n][attribute] = str(res[attribute])
                    run_n += 1
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
            
    def single_optimizer_experiment(conf_id, databatch_id, data_type, num_data_points, s_rank, unitary, data_points):
        result_dict = {}
        data_points_string = (
            np.array2string(data_points.numpy(), separator=",")
            .replace("\n", "")
            .replace(" ", "")
        )
        result_dict["databatch"] = data_points_string
        
        # specifications of qnn
        qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
        
        expected_output = torch.matmul(unitary, data_points)
        y_true = expected_output.conj()
        
        # objective function based on cost function of qnn 
        def objective(x):
            qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
            cost = cost_func(data_points, y_true, qnn, device="cpu") 
            return torch.tensor(cost.item())

        # verschiedene inital_param_values ausprobieren und avg bilden? 
        initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) # [0,2pi] siehe victor_thesis_landscapes.py, bei allen optimierern gleich
        initial_param_values_tensor = torch.tensor(initial_param_values)

        # run optimizer experiments
        sgd_optimizers = ['sgd', 'rmsprop', 'adam']
        optimizers = [nelder_mead_experiment, bfgs_experiment, cobyla_experiment, powell_experiment, slsqp_experiment, sgd_experiment]

        with ProcessPoolExecutor(cpu_count()) as exe:
            for opt in optimizers:
                if opt == sgd_experiment:
                    for variant in sgd_optimizers:
                        future = exe.submit(sgd_experiment, objective, initial_param_values_tensor, variant)
                        opt_name = variant
                        result_dict[opt_name] = future.result()
                else:
                    future = exe.submit(opt, objective, initial_param_values)
                    opt_name = opt.__name__.removesuffix('_experiment')
                    result_dict[opt_name] = future.result()

        return result_dict
