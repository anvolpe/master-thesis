import csv
import gc
import json
import time
from datetime import datetime
from qnns.cuda_qnn import CudaPennylane
import re

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
from scipy.optimize import minimize, dual_annealing
import re

def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            print(f"Lade Datei: {file_path}")
            with open(file_path, 'r') as file:
                try:
                    json_data = json.load(file)
                    data.append(json_data)
                except json.JSONDecodeError:
                    print(f"Fehler beim Laden der Datei: {file_path}")
    if not data:
        print("Keine JSON-Dateien gefunden oder alle Dateien sind fehlerhaft.")
    return data

def extract_solution_x_data(json_data):
    '''
        Extract mean x_min and x_max value for every optimizer for every bound for one config.
    '''
    gradient_free = ["nelder_mead", "powell", "cobyla"]
    gradient_based = ["sgd", "adam", "rmsprop", "bfgs", "dual_annealing","slsqp"]
    optimizers = gradient_based + gradient_free
    bounds_batches = ["bounds_0", "bounds_1", "bounds_2", "bounds_3", "bounds_4"]
    databatches = ["databatch_0", "databatch_1", "databatch_2", "databatch_3", "databatch_4"]
    optimizer_data = {}
    #gradient_based_data = []
    #gradient_free_data = []

    # prepare results dict
    # bounds_i : opt_1 : [(x_min1, x_max1), (x_min2, x_max2),...], opt_2 : ...
    res_min = {"bounds_0": {}, "bounds_1": {}, "bounds_2": {}, "bounds_3": {}, "bounds_4": {}}
    res_max = {"bounds_0": {}, "bounds_1": {}, "bounds_2": {}, "bounds_3": {}, "bounds_4": {}}

    for i in range(len(json_data)):
        print(f"Verarbeite config_{i}")
        for databatch_id in databatches:
            print(f"Verarbeite {databatch_id}")
            for bounds_id in bounds_batches:
                print(f"Verarbeite {bounds_id}")
                for opt in optimizers:
                    print(f"Verarbeite {opt}")
                    try:
                        dict = json_data[i][databatch_id][bounds_id][opt]["0"]
                        if opt not in res_min[bounds_id]:
                            res_min[bounds_id][opt] = []
                        if opt not in res_max[bounds_id]:
                            res_max[bounds_id][opt] = []
                        x = dict["x"]
                        x = [float(idx) for idx in x.strip('[ ]').split()]
                        res_min[bounds_id][opt].append(np.min(x))
                        res_max[bounds_id][opt].append(np.max(x))
                    except KeyError as e:
                        print(f"Fehler beim Lesen der Daten: {e}")

    # Berechne Pro Optimierer (pro Bounds) untere x-Grenze und obere x-Grenze
    res_min_max = {"bounds_0": {}, "bounds_1": {}, "bounds_2": {}, "bounds_3": {}, "bounds_4": {}}
    for bounds_id in bounds_batches:
        print(f"Verarbeite {bounds_id}")
        for opt in optimizers:
            print(f"Verarbeite {opt}")
            try:
                res_min_max[bounds_id][opt] = (np.min(res_min[bounds_id][opt]),np.max(res_max[bounds_id][opt]))
            except KeyError:
                print(f'Optimierer existiert für diese bounds nicht.')
    
    return res_min, res_max, res_min_max

def create_min_max_boxplots(res_min, res_max, save_path):
    bounds = {"bounds_0": "No Bounds", "bounds_1": r"$[0, 2\pi]$", "bounds_2": r"$[0, 4\pi]$", "bounds_3": r"$[-2\pi, 2\pi]$", "bounds_4": r"$[-4\pi, 4\pi]$"}
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for bounds_id in bounds.keys():
        plt.figure(figsize=(10,10))
        data_min = res_min[bounds_id]
        data_max = res_max[bounds_id]
        x = np.array([(i+1)*1000 for i in range(len(data_min.keys()))])
        plt.boxplot(data_min.values(), sym="", vert=False,positions=x-200,widths=200)
        plt.boxplot(data_max.values(), sym="", vert=False,positions=x+200,widths=200)
        plt.yticks(ticks=x,labels=data_min.keys())
        plt.ylabel('Optimizer')
        plt.xlabel('Minimal (lower) and maximal (upper) x-values')
        plt.title(f"Minimal and Maximal x-Values for bounds: {bounds[bounds_id]}",fontsize='xx-large')
        #plt.legend()
        #plt.ylim(bottom=-0.015,  top=max(fun) + 0.05)
        plt.grid(True)
        file_path = os.path.join(save_path, f'{bounds_id}_boxplot_no_outliers.png')
        plt.savefig(file_path)
        plt.close()



def boxplot_fun_values_per_optimizers():
    #TODO: read conf_[conf_id]_opt.json files: all fun values + optimizer
    #one violin plot (boxplot??) per optimizer
    print("TODO")

if __name__ == "__main__":
    path = "experimental_results/results/optimizer_results/bounds/"
    data = load_json_files(path)
    print(data[0]["conf_id"])
    res_min,res_max,res = extract_solution_x_data(data)
    create_min_max_boxplots(res_min, res_max, 'qnn-experiments/experimental_results/results/box_plots/bounds/no_outliers')