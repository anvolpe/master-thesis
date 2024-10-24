import csv
import time
from datetime import datetime
from qnns.cuda_qnn import CudaPennylane

from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from victor_thesis_experiments_main import *

from sgd_for_scipy import *
from scipy.optimize import minimize, dual_annealing
import pyswarms as ps

import numpy as np
import torch
from scipy.optimize import minimize


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



# Callback: Save every 10th intermediate results of each optimization
fun_all = [] # array for callback function (save every 10th fun value during optimization)
nit = 0
def saveIntermResult(intermediate_result: OptimizeResult):
    fun=intermediate_result.fun
    global nit
    if(nit%10==0):
        fun_all.append(float(fun))
    nit += 1

# Callback variant: Save EVERY intermediate result (for Powell and BFGS)
def saveIntermResultEvery(intermediate_result: OptimizeResult):
    fun=intermediate_result.fun
    global nit
    fun_all.append(float(fun))
    nit += 1

#create individual callback for specific objective function. objectivew function is the used to calculate iterm Result
def getCallback(objective_func):
#use signature with xk as current Vector and CALCulate interm Result
#for methods that dont support OptimizeResult Signature (slsqp, cobyla)
    def saveIntermResult_Calc(xk):
        fun=objective_func(xk)
        global nit
        if(nit%10==0):
            fun_all.append(float(fun))
        nit += 1
    return saveIntermResult_Calc

#use specific callback Signature for dual annealing
#(x,f,context) with f being the current function value
#WARNING: Every function value is saved, not just every 10th function value
def saveIntermResult_duAn(x, f, context):
    fun=f
    global nit
    fun_all.append(float(fun))
    nit +=1 

def nelder_mead_experiment(objective,initial_param_values,bounds=None):
    results = {"type": "gradient-free"}
    run_n = 0
    for max_iter in max_iters:
        for fatol in tols:
            for xatol in tols:
                start = time.time()
                res = minimize(objective, initial_param_values, method="Nelder-Mead", bounds=bounds, callback=saveIntermResult,
                        options={"maxiter": max_iter, "fatol":fatol, "xatol":xatol})
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "fatol":fatol, "xatol":xatol, "duration":duration}
                # result info
                for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])
                #global fun_all
                results[run_n]["callback"] = list(fun_all)
                fun_all.clear()
                global nit 
                nit = 0
                run_n += 1
    return results

# callback not supported
def cobyla_experiment(objective,initial_param_values,bounds=None):
    results = {"type" : "gradient-free"}
    run_n = 0
    for max_iter in max_iters:
        for tol in tols:
            for catol in tols:
                temp_callback=getCallback(objective_func=objective)
                start = time.time()
                res = minimize(objective, initial_param_values, method="COBYLA", bounds=bounds,  
                        options={"maxiter": max_iter, "tol":tol, "catol":catol}, callback=temp_callback)
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "tol":tol, "catol":catol, "duration":duration}
                # result info
                for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])
                results[run_n]["callback"] = list(fun_all)
                fun_all.clear()
                global nit 
                nit = 0
                run_n += 1
    return results

def bfgs_experiment(objective,initial_param_values,bounds=None):
    results = {"type": "gradient"}
    run_n = 0
    for max_iter in max_iters:
        for gtol in tols:
            for xrtol in tols:
                for eps in tols:
                    start = time.time()
                    res = minimize(objective, initial_param_values, method="BFGS", bounds=bounds,  callback=saveIntermResultEvery,
                            options={"maxiter": max_iter, "gtol":gtol, "xrtol":xrtol, "eps":eps})
                    duration = time.time() - start
                    # fill results dict
                    # specifications of this optimizer run
                    results[run_n] = {"maxiter": max_iter, "gtol":gtol, "xrtol":xrtol, "eps":eps, "duration":duration}
                    # result info
                    for attribute in res.keys():
                        results[run_n][attribute] = str(res[attribute])
                    results[run_n]["callback"] = list(fun_all)
                    fun_all.clear()
                    global nit 
                    nit = 0
                    run_n += 1
    return results

def powell_experiment(objective,initial_param_values,bounds=None):
    results = {"type": "gradient-free"} # TODO: stimmt das??
    run_n = 0
    for max_iter in max_iters:
        for ftol in tols:
            for xtol in tols:
                start = time.time()
                res = minimize(objective, initial_param_values, method="Powell", bounds=bounds, callback=saveIntermResultEvery,
                        options={"maxiter": max_iter, "ftol":ftol, "xtol":xtol})
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "ftol":ftol, "xtol":xtol, "duration":duration}
                # result info
                for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])
                results[run_n]["callback"] = list(fun_all)
                fun_all.clear()
                global nit 
                nit = 0
                run_n += 1
    return results

def slsqp_experiment(objective,initial_param_values,bounds=None):
    results = {"type": "gradient"} #TODO: stimmt das?
    run_n = 0
    for max_iter in max_iters:
        for ftol in tols:
            for eps in tols:
                temp_callback=getCallback(objective_func=objective)
                start = time.time()
                res = minimize(objective, initial_param_values, method="SLSQP", bounds=bounds,  
                        options={"maxiter": max_iter, "ftol":ftol, "eps":eps}, callback=temp_callback)
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "ftol":ftol, "eps":eps, "duration":duration}
                # result info
                for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])
                results[run_n]["callback"] = list(fun_all)
                fun_all.clear()
                global nit 
                nit = 0
                run_n += 1
    return results

def sgd_experiment(objective,initial_param_values,opt,bounds=None):
    results = {"type": "gradient"}
    run_n = 0
    for max_iter in max_iters:
        for learning_rate in learning_rates:
            for eps in tols:
                start = time.time()
                res = minimize(objective, initial_param_values, method=opt,  callback=saveIntermResult,
                        options={"maxiter": max_iter, "learning_rate":learning_rate, "eps":eps})
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "learning_rate":learning_rate, "eps":eps, "duration":duration}
                # result info
                for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])
                results[run_n]["callback"] = list(fun_all)
                fun_all.clear()
                global nit 
                nit = 0
                run_n += 1
    return results

def dual_annealing_experiment(objective,initial_param_values,bounds=default_bounds):
    results = {"type": "gradient-free"} 
    run_n = 0
    for max_iter in max_iters:
        #for tol in tols:
        #for catol in tols:
                start = time.time()
                res = dual_annealing(objective, bounds, maxiter=max_iter, callback=saveIntermResult_duAn)
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "duration":duration}
                # result info
                for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])
                results[run_n]["callback"] = list(fun_all)
                #print("es folgen die funktionswerte von dual annealing")
                #print(fun_all)
                fun_all.clear()
                global nit 
                nit = 0
                run_n += 1
    return results

def particle_swarm_experiment(objective, initial_param_values,bounds=None):
    results = {"type": "gradient-free"} 
    run_n = 0
    # dimensions in particle swarm = number of parameters in QNN, i.e. length of initial_param_values, i.e. dimensions = 6 (see line 26)
    # Set-up hyperparameters
    # c1 = cognititve parameter, c2 = social parameter, w = controls inertia of swarm movement
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    run_n = 0
    for max_iter in max_iters:
        #for tol in tols:
        #for catol in tols:
                # Call instance of PSO
                optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=dimensions, options=options)

                # Perform optimization
                start = time.time()
                res, pos = optimizer.optimize(objective, iters=max_iter)
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "duration":duration}
                # result info
                '''for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])'''
                results[run_n]["fun"] = res
                results[run_n]["x"] = pos
                results[run_n]["ftol"] = optimizer.ftol
                results[run_n]["callback"] = list(optimizer.cost_history) # stimmt das??
                
                global nit 
                nit = 0
                run_n += 1
    return results




#Ideal hyperparameter for 6-dimensional problems according to **(Quelle.../ keine Angaben zu SM und Elitism/)** 
#Population Size:20-100,Mutation Rate:0,01-0, Crossover Rate:0,7-0.9,  Number of generations:50-200,(Selection Method: not clear),(Elitism: %-% not clear) 

#def genetic_algorithm_experiment(objective, initial_param_values, bounds=None):
    
    pop_size = 20
    mutation_rate = 0.1
    crossover_rate = 0.7
    num_generations = 50
    
    # Initialize results dictionary
    results = {"type": "gradient-free"}
    run_n = 0
    
    def fitness(individual):
        return objective(individual)
    
    def initialize_population():
        population = []
        for _ in range(pop_size):
            individual = np.random.uniform(low=bounds[0][0], high=bounds[0][1], size=len(initial_param_values))
            population.append(individual)
        return np.array(population)
    
    def crossover(parent1, parent2):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, len(initial_param_values))
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        else:
            return parent1, parent2
    
    def mutate(individual):
        for i in range(len(initial_param_values)):
            if np.random.rand() < mutation_rate:
                individual[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        return individual

    population = initialize_population()
    best_individual = None
    best_fitness = float('inf')
    #Track the fittest over generations
    fitness_evolution = []  
    
    for generation in range(num_generations):
        fitness_values = np.array([fitness(ind) for ind in population])
        sorted_idx = np.argsort(fitness_values)
        population = population[sorted_idx]
        fitness_values = fitness_values[sorted_idx]
        
        if fitness_values[0] < best_fitness:
            best_fitness = fitness_values[0]
            best_individual = population[0]
        
        # Keep fittest for each generation
        fitness_evolution.append(best_fitness)
        
        # Carry over the two fittest 
        next_generation = population[:2]
        
        # Generate new population
        while len(next_generation) < pop_size:
            parent1 = population[np.random.randint(0, pop_size // 2)]
            parent2 = population[np.random.randint(0, pop_size // 2)]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            next_generation = np.vstack((next_generation, child1, child2))
        
        population = next_generation[:pop_size]
        print(f"Generation {generation}: Best fitness = {best_fitness}")

    results[run_n] = {
        "maxiter": num_generations,
        "best_fitness": best_fitness,
        "best_individual": best_individual.tolist(),
        "fitness_evolution": fitness_evolution
    }
    
    return results

    
    