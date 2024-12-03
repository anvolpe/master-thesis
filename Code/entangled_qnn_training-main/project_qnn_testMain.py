import itertools
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
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import re

import pandas as pd

opt_titles = {'nelder_mead': 'Nelder-Mead', 'powell':'Powell', 'sgd':'SGD', 
              'adam':'Adam', 'rmsprop':'RMSprop', 'bfgs':'BFGS','slsqp':'SLSQP',
              'dual_annealing':'Dual Annealing','cobyla':'COBYLA',
              'genetic_algorithm':'Genetic Algorithm', 'particle_swarm': 'Particle Swarm Optimization',
              'diff_evolution':'Differential Evolution'}

opt_info = {"nelder_mead": "maxiter (100, 500, 1000), fatol (1e-5, 1e-10), and xatol (1e-5, 1e-10)", 
            "powell": "maxiter (50, 100, 1000), ftol (1e-5, 1e-10) and xtol (1e-5, 1e-10)", 
            "sgd":"maxiter (50, 100, 1000), learning_rate (0.01, 0.001, 0.0001) and eps (1e-5, 1e-10)", 
              "adam":"maxiter (50, 100, 1000), learning_rate (0.01, 0.001, 0.0001) and eps (1e-5, 1e-10)", 
              "rmsprop":"maxiter (50, 100, 1000), learning_rate (0.01, 0.001, 0.0001) and eps (1e-5, 1e-10)", 
              "bfgs":"maxiter (50, 100, 1000), gtol (1e-5, 1e-10), xrtol (1e-5, 1e-10) and eps (1e-5, 1e-10)",
              "slsqp":"maxiter (50, 100, 1000), ftol (1e-5, 1e-10) and eps (1e-5, 1e-10)",
              "dual_annealing":"maxiter (50, 100, 1000)",
              "cobyla":"maxiter (50, 100, 1000), tol (1e-5, 1e-10) and catol (1e-5, 1e-10)",
              "genetic_algorithm":"maxiter (50, 100, 1000), crossover_type (single_point, two_points, uniform, scattered), stop_criteria (None, saturate_50)", 
              "particle_swarm": "maxiter (100, 500, 1000), ftol (1e-5, -np.Infinity)",
              "diff_evolution":"maxiter (100, 500, 1000)"}

db_list = [f"databatch_{i}" for i in range(0,5)]
def func1():
    time.sleep(3)
    print("done with func1")
    return None

def func2():
    time.sleep(4)
    print("done with func2")
    return None

def pso_test():
    num_layers = 1
    num_qubits = 2
    dimensions = 6
    max_iters = [100,500,1000]
        # setup qnn configuration
    conf_id = 0
    data_type = "random" # random, orthogonal, non_lin_ind, var_s_rank
    num_data_points = 1
    s_rank = 1 # Schmidt-Rank
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


    # setup qnn
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()
    
    # objective function based on cost function of qnn 
    def objective(x):
        '''
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
    results = {}
    # verschiedene inital_param_values ausprobieren und avg bilden? 
    # TODO: gleich für alle Konfigs oder nur gleich für eine Konfig aber da für alle Optmimierer?
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) # [0,2pi] siehe victor_thesis_landscapes.py, bei allen optimierern gleich
    initial_param_values_tensor = torch.tensor(initial_param_values)

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
                results[run_n]["callback"] = list(optimizer.cost_history) # stimmt das??
                
                global nit 
                nit = 0
                run_n += 1
    print(results)

def change_s_rank(directory):
    '''
        Corrects Schmidt Rank in a JSON file depending on its config id. 
    '''
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            #print(f"Lade Datei: {file_path}")
            data = []
            with open(file_path, 'r') as file:
                data = json.load(file)
                id = int(data["conf_id"])
                s_rank = int((id % 20)/5)+1
                data["s_rank"] = s_rank

            with open(file_path, 'w') as file: 
                json.dump(data, file, indent=4)
            del data

def add_opt_info(directory):
    '''
        Corrects Schmidt Rank in a JSON file depending on its config id. 
    '''
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            #print(f"Lade Datei: {file_path}")
            data = []
            with open(file_path, 'r') as file:
                data = json.load(file)
            for databatch in db_list:
                for opt in opt_titles.keys():
                    try:
                        d = data[databatch][opt]
                        new_d = {"type": d.pop("type"), "variable hyperparams": opt_info[opt], **d}
                        data[databatch][opt] = new_d
                        del d
                        del new_d
                    except KeyError as e:
                        #continue
                        print(f"Schlüssel fehlt: {e}")

            with open(file_path, 'w') as file: 
                json.dump(data, file, indent=4)
            del data

def change_pso_json(directory):
    '''
        Corrects Schmidt Rank in a JSON file depending on its config id. 
    '''
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            #print(f"Lade Datei: {file_path}")
            data = []
            with open(file_path, 'r') as file:
                data = json.load(file)
            print(data["conf_id"])
            for databatch in db_list:
                try:
                    opt = "particle_swarm"
                    d = data[databatch][opt]
                    d.pop("hyperparameters")
                    new_d = {"type": "gradient-free", "variable hyperparams": opt_info[opt], **d}
                    data[databatch][opt] = new_d
                    del d
                    del new_d
                except KeyError as e:
                    #continue
                    print(f"Schlüssel fehlt: {e}")

            with open(file_path, 'w') as file: 
                json.dump(data, file, indent=4)
            del data

if __name__ == "__main__":
    os.chdir("../../")
    directory = "experimental_results/results/optimizer_results/experiment_part2_GA_PSO_DE"
    #directory = "experimental_results/results/optimizer_results"
    change_s_rank(directory)
    print("s_rank done")
    add_opt_info(directory)
    print("add opt info done")
    change_pso_json(directory)
    print("pso done")    
    #print(db_list)


   





    
    