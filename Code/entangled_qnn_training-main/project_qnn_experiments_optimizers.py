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
import pygad as pg


#no_of_runs = 1
no_of_runs = 10

num_layers = 1
num_qubits = 2
dimensions = 6
max_iters = [100,500,1000]
#max_iters = [1000]
tols = [1e-10]
#tols = [1e-5, 1e-10]
#tols = [1e-10, 1e-15] schlechte ergebnisse, 1e-5 viel besser
#tols = [1e-2, 1e-5]
#bounds = [(0,2*np.pi)*dimensions]
default_bounds = list(zip(np.zeros(6), np.ones(6)*2*np.pi))
#bounds = list(zip(np.ones(6)*(-2)*np.pi, np.ones(6)*2*np.pi))
learning_rates = [0.01, 0.001, 0.0001]
#learning_rates = [0.0001]

# hyperparameters for PSO
swarm_sizes = [10,30] #n_particles
inertia_values = [0.5, 0.9] #w
cognitive_social_value_pairs = [[0.5, 0.5], [0.5, 2], [2, 0.5]] # c1, c2
#cognitive_social_value_pairs = [[0.5, 0.3]]


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

def particle_swarm_experiment(objective,bounds=None):
    results = {"type": "gradient-free"} 
    results = {"hyperparameters": f"max_iter: {max_iters}, n_particles (swarm size): {swarm_sizes}, w (inertia): {inertia_values}, [c1,c2] (cognitive and social parameter): {cognitive_social_value_pairs}."}
    run_n = 0
    # dimensions in particle swarm = number of parameters in QNN, i.e. length of initial_param_values, i.e. dimensions = 6 (see line 26)
    # Set-up hyperparameters
    # c1 = cognititve parameter, c2 = social parameter, w = controls inertia of swarm movement
    
    run_n = 0
    for max_iter in max_iters:
    #for max_iter in [1000]:
        for S in swarm_sizes:
            for w in inertia_values:
                for c1,c2 in cognitive_social_value_pairs:
                    for tol in tols:
                        options = {'c1': c1, 'c2': c2, 'w':w}
                        # Call instance of PSO
                        optimizer = ps.single.GlobalBestPSO(n_particles=S, dimensions=dimensions, options=options, ftol=tol, ftol_iter=25)

                        # Perform optimization
                        start = time.time()
                        res, pos = optimizer.optimize(objective, iters=max_iter)
                        duration = time.time() - start
                        # fill results dict
                        # specifications of this optimizer run
                        results[run_n] = {"maxiter": max_iter, "duration":duration}
                        # result info
                        results[run_n]["max_iter"] = max_iter
                        results[run_n]["n_particles"] = S
                        results[run_n]["c1"] = c1
                        results[run_n]["c2"] = c2
                        results[run_n]["w"] = w
                        results[run_n]["fun"] = res
                        results[run_n]["x"] = list(pos)
                        results[run_n]["nit"] = len(optimizer.cost_history)
                        results[run_n]["ftol"] = optimizer.ftol
                        results[run_n]["ftol_iter"] = optimizer.ftol_iter
                        results[run_n]["callback"] = list(optimizer.cost_history)
                        run_n += 1
    return results


# GA von Alina
# TODO: Hyperparameter einstellen: Anzahl Generationen, Eltern, init_range, parent selection, crossover, mutation, etc.
# TODO: Hyperparameter und richtige Ergebnisse in results dictionary speichern.

selection_type_list = ["sss", "rws", "tournament", "rank"]
crossover_type_list = ["single_point", "two_points", "uniform"]
mutation_type_list = ["random", "swap"]
max_gens = [100, 500, 1000]


def genetic_algorithm_experiment(objective, bounds=None):
    results = {"type": "gradient-free"} 
    #results = {"hyperparameters": f"max_iter: {max_iters}, n_particles (swarm size): {swarm_sizes}, w (inertia): {inertia_values}, [c1,c2] (cognitive and social parameter): {cognitive_social_value_pairs}."}
    run_n = 0
    # fitness function: negative cost function value (since higher fitness means better solution --> instead of minimizing, GA maximizes)
    def fitness(ga_instance, solution, solution_idx):
        return -objective(solution)
    
    #potential starting parameters, as suggested by PyGAD
    fitness_function = fitness

    #num_generations = 50 # TODO: vergleichbar mit max_iter? oder sind da Werte für 100, 500 und 1000 zu hoch?
    num_parents_mating = 4

    sol_per_pop = 8
    num_genes = dimensions

    #init_range_low = -2
    #init_range_high = 5

    #parent_selection_type = "sss" # selection method?
    keep_parents = 1

    #crossover_type = "single_point"

    #mutation_type = "random" 
    mutation_percent_genes = 10

    for num_generations in max_gens:
        for parent_selection_type in selection_type_list:
            for crossover_type in crossover_type_list:
                for mutation_type in mutation_type_list:
                    ga_instance = pg.GA(num_generations=num_generations,
                                    num_parents_mating=num_parents_mating,
                                    fitness_func=fitness_function,
                                    sol_per_pop=sol_per_pop,
                                    num_genes=num_genes,
                                    parent_selection_type=parent_selection_type,
                                    keep_parents=keep_parents,
                                    crossover_type=crossover_type,
                                    mutation_type=mutation_type,
                                    mutation_percent_genes=mutation_percent_genes)
                    start = time.time()
                    ga_instance.run()
                    duration = time.time() - start

                    print(f"{run_n} done: {duration} sek")

                    solution, solution_fitness, solution_idx = ga_instance.best_solution()
                    results[run_n] = {"name": "test GA"}
                    results[run_n]["fun"] = -solution_fitness
                    results[run_n]["x"] = list(solution)
                    results[run_n]["ngeneration"] = int(ga_instance.best_solution_generation)
                    results[run_n]["max_generation"] = num_generations
                    results[run_n]["num_parents_mating"] = num_parents_mating
                    results[run_n]["sol_per_pop"] = sol_per_pop
                    results[run_n]["parent_selection_type"] = parent_selection_type
                    results[run_n]["keep_parents"] = keep_parents
                    results[run_n]["crossover_type"] = crossover_type 
                    results[run_n]["mutation_type"] = mutation_type
                    results[run_n]["mutations_percent_genes"] = mutation_percent_genes 
                    results[run_n]["callback"] = list(ga_instance.best_solutions_fitness) # TODO: Einzelne Einträge negieren um die eigentlichen Werte zu kriegen
                    run_n += 1
    return results