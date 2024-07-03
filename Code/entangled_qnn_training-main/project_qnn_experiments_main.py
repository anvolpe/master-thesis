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
import os
import pandas as pd
from scipy.optimize import minimize


def optimizer_experiment():
    # specifications of qnn
    num_layers = 1 # 2??
    num_qubits = 2 # 1??
    dimensions = 6 # geht nicht: alles außer 6
    type_of_data = 4 # random data
    deg_of_entanglement = 1 # high, (low = 1)
    num_data_points = 1 # low

    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    
    # generate a random unitary with num_qubits qubits (why are they the same?)
    unitary = torch.tensor(
            np.array(random_unitary_matrix(num_qubits)),
            dtype=torch.complex128,
            device="cpu",
        )
    data_points = generate_data_points(
        type_of_data,
        deg_of_entanglement,
        num_data_points,
        unitary, num_qubits
    )
    print(data_points.shape)
    print(data_points)
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()
    
    # objective function based on cost function of qnn 
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return cost.item()
    
    # 
    optimizers = ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP']
    results = {}
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions)
    for opt in optimizers:
        start = time.time()
        res = minimize(objective, initial_param_values, method=opt,
                       options={"maxiter": 100})
        duration = time.time() - start
        results[opt] = {'result': res, 'duration': duration}
        print(f"Optimizer: {opt}")
        print(res)
        print(f"Duration: {duration}s\n")

    print("Results:", results)

    with open('experimental_results/results/optimization_results.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["Optimizer", "Result", "Duration"])

        for opt, result in results.items():
            writer.writerow([opt, result['result'], result['duration']])



if __name__ == "__main__":
    optimizer_experiment()