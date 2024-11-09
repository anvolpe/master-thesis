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

import numpy as np
import random
import torch
from scipy.optimize import minimize
import time

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
    
    pop_size = 20  
    mutation_rate = 0.1
    crossover_rate = 0.7
    num_generations = 50

    def objective(self, x):
        self.qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(self.qnn.params.shape) 
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
                    # Store results for the current optimization run
                    # Specifications for this optimizer run
                    results[run_n] = {"maxiter": max_iter, "fatol":fatol, "xatol":xatol, "duration":duration}
                    # Store the result information
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
                    # Store results for the current optimization run
                    # Specifications for this optimizer run
                    results[run_n] = {"maxiter": max_iter, "tol":tol, "catol":catol, "duration":duration}
                    # Store the result information
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
                        # Store results for the current optimization run
                        # Specifications for this optimizer run
                        results[run_n] = {"maxiter": max_iter, "gtol":gtol, "xrtol":xrtol, "eps":eps, "duration":duration}
                        # Store the result information
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
                    # Store results for the current optimization run
                    # Specifications for this optimizer run
                    results[run_n] = {"maxiter": max_iter, "ftol":ftol, "xtol":xtol, "duration":duration}
                    # Store the result information
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
                    # Store results for the current optimization run
                    # Specifications for this optimizer run
                    results[run_n] = {"maxiter": max_iter, "ftol":ftol, "eps":eps, "duration":duration}
                    # Store the result information
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
                    # Store results for the current optimization run
                    # Specifications for this optimizer run
                    results[run_n] = {"maxiter": max_iter, "learning_rate":learning_rate, "eps":eps, "duration":duration}
                    # Store the result information
                    for attribute in res.keys():
                        results[run_n][attribute] = str(res[attribute])
                    run_n += 1
        return results
    
    def genetic_algorithm_experiment(self, conf_id, data_type, num_data_points, s_rank, unitary, data_points):

        self.data_points = data_points
        self.y_true = torch.matmul(unitary, data_points).conj()

        def fitness(individual):
            return self.objective(individual).item()

        def initialize_population():
            population = []
            for _ in range(self.pop_size):
                individual = np.random.uniform(low=self.bounds[0][0], high=self.bounds[0][1], size=self.dimensions)
                population.append(individual)
            return np.array(population)

        def crossover(parent1, parent2):
            if np.random.rand() < self.crossover_rate:
                # Select a random point between 1 and the number of dimensions
                crossover_point = np.random.randint(1, self.dimensions)
                 # Combine genes from parent1 up to the crossover point and genes from parent2 after the crossover point
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                return child1, child2
            else:
                return parent1, parent2

        def mutate(individual):
            # Iterate over all dimensions and check if mutation occurs for each dimension
            for i in range(self.dimensions):
                if np.random.rand() < self.mutation_rate:
                    individual[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
            return individual

        # Main Loop
        population = initialize_population()
        best_individual = None
        best_fitness = float('inf')

        for generation in range(self.num_generations):
            fitness_values = np.array([fitness(ind) for ind in population])
            sorted_idx = np.argsort(fitness_values)
            population = population[sorted_idx]
            fitness_values = fitness_values[sorted_idx]

            if fitness_values[0] < best_fitness:
                best_fitness = fitness_values[0]
                best_individual = population[0]

           # Keep the top 2 individuals for elitism (ensure the best individuals from each generation are carried over unchanged to the next generation)
            next_generation = population[:2] 

            while len(next_generation) < self.pop_size:
                parent1, parent2 = population[np.random.randint(0, self.pop_size // 2)], population[np.random.randint(0, self.pop_size // 2)]
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1)
                child2 = mutate(child2)
                next_generation = np.vstack((next_generation, child1, child2))

            population = next_generation[:self.pop_size]

            print(f"Generation {generation}: Best fitness = {best_fitness}")

        #Save results 
        os.makedirs("experimental_results/results/optimizer_results", exist_ok=True)
        with open(f"experimental_results/results/optimizer_results/conf_{conf_id}_ga.csv", mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Best_Fitness", "Best_Individual"])
            writer.writerow([generation, best_fitness, best_individual])

        return {"best_individual": best_individual, "best_fitness": best_fitness}
    
    

    def single_config_experiments(conf_id, data_type, num_data_points, s_rank, unitary, data_points):
        # Prepare csv file for experiment results
        os.makedirs("experimental_results/results/optimizer_results", exist_ok=True)
        file = open(f"experimental_results/results/optimizer_results/conf_{conf_id}_opt.csv", mode="w")
        writer = csv.writer(file)
        writer.writerow(["Optimizer", "Result", "Duration"])
        
        # Specifications of qnn
        qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
            
        expected_output = torch.matmul(unitary, data_points)
        y_true = expected_output.conj()

         # Objective function based on the qnn's cost function
         # Set qnn parameters with current optimizer values(reshaping if necessary)
        def objective(x):
            qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) 
            cost = cost_func(data_points, y_true, qnn, device="cpu") 
            return torch.tensor(cost.item())
       
        # Set identical parameter values for all optimizers
        # Range is [0, 2π] as per Victor's thesis; these are consistent across optimizers
        initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions)
        initial_param_values_tensor = torch.tensor(initial_param_values) 

         # Run all optimizer experiments
         # Each experiment returns a result dictionary that includes all specifications
         # Save this result as a CSV file named conf_[conf_id]_opt.csv
            
    def single_optimizer_experiment(conf_id, databatch_id, data_type, num_data_points, s_rank, unitary, data_points):
        result_dict = {}
        data_points_string = (
            np.array2string(data_points.numpy(), separator=",")
            .replace("\n", "")
            .replace(" ", "")
        )
        result_dict["databatch"] = data_points_string
        
        # Set up the qnn with specified parameters
        qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
        
        
        expected_output = torch.matmul(unitary, data_points)
        y_true = expected_output.conj() 

        # Objective function based on the qnn's cost function
        def objective(x):
            # Update QNN parameters with the current values from the optimizer
            qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape)
            cost = cost_func(data_points, y_true, qnn, device="cpu")
            return torch.tensor(cost.item())

        # Generate initial parameter values within the range [0, 2π] for all optimizers
        initial_param_values = np.random.uniform(0, 2 * np.pi, size=dimensions)
        initial_param_values_tensor = torch.tensor(initial_param_values)

        # Run optimizer experiments
        sgd_optimizers = ['sgd', 'rmsprop', 'adam']
        optimizers = [nelder_mead_experiment, bfgs_experiment, cobyla_experiment,powell_experiment, slsqp_experiment, sgd_experiment]

        # Parallel execution of optimizers
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
    