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

def generate_and_save_testLandscape():
    print("####### Generate and Save test landscape")
    num_layers = 1
    num_qubits = 2
    #num_unitaries = 1 # was 5
    #num_tries = 1 # was 5
    grid_size = 16
    dimensions = 6 # geht nicht: alles außer 6
    type_of_data = 1 # random data
    deg_of_entanglement = 4 # low, (high = 4)
    num_data_points = 1 # low

    # generate an experiment id (based on time) to identify which results and configs belong to which experiment run
    current_time = datetime.now()
    exp_id = (
        str(grid_size)
        + "_"
        + str(dimensions)
        + "_"
        + str(current_time.month)
        + "_"
        + str(current_time.day)
        + "_"
        + str(current_time.hour)
        + "_"
        + str(current_time.minute)
        + "_"
        + str(current_time.second)
    )
    # create directories for results and configs

    #os.makedirs("experimental_results/configs", exist_ok=True)
    #os.makedirs("experimental_results/results", exist_ok=True)
    # generate a U3 ansatz containing 2 layers -> 6 params
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    #unitaries = []
    # [type_of_data][num_data_points][deg_of_entanglement][id_unitary][id_try]
    #configurations = []
    
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
    #start = time.time()
    # generate configurations (5 datapoint sets = 5 runs per config)
    conf_id = 0

    TV_arr = []
    FD_arr = []
    IGSD_arr = []
    SC_metrics = []
    start = time.time()
    landscape = generate_loss_landscape(grid_size, dimensions, data_points, unitary, qnn)
    t = time.time()-start
    print("time [generate loss landscape] (sec): ", np.round(t,2))
    start = time.time()
    TV_arr.append(calc_total_variation(landscape))
    FD_arr.append(calc_fourier_density(landscape))
    IGSD_arr.append(calc_IGSD(landscape))
    SC = calc_scalar_curvature(landscape)
    SC_metrics.append(process_sc_metrics(SC))
    t = time.time()-start
    print("time [calc metrics] (sec): ", np.round(t,2))
    del SC
    # save landscape as JSON file
    os.makedirs("experimental_results/landscapes", exist_ok=True)
    start = time.time()
    np.savez_compressed("experimental_results/landscapes/testLandscape.npz", np.array(grid_size),landscape)
    t = time.time()-start
    print("time [write npz file] (sec): ", np.round(t,2))

    gc.collect() # garbage collector
        
    metrics = []
    metrics.append(TV_arr)
    metrics.append(FD_arr)
    metrics.append(IGSD_arr)
    metrics.append(SC_metrics)
    process_and_store_metrics(metrics, 1, conf_id, exp_id)
    # TODO: speichern: Optimierer, Zeiten, Spezifikation
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"[{now}] Finished run: {conf_id}")

def run_single_optimizer_experiment():
    # TODO: def objective basierend auf landscape
    # TODO: optimize.minimize(...) Aufruf für einen Optimierer + Options (Spezifikationen)
    # TODO: Zeit messen + Zeit speichern

    # load test loss landscape from npz file (67mb)
    start = time.time()
    landscapeZ = np.load("experimental_results/landscapes/testLandscape.npz") # falls 
    grid_size = landscapeZ['arr_0']
    landscape = landscapeZ['arr_1']
    t = time.time()-start
    print("time [load npz file] (sec): ", np.round(t,2))

    # define objective basd on loss landscape and grid_size
    # input x: is a 1D array with 6 values (since: 6 parameters)
    # output: corresponding loss 
    def objective(x):
        '''
            objective to minimize based on loss landscape

            Args:
                x (1D array): is a 1D array with 6 values (since: 6 parameters)
            Returns:
                float: corresponding loss 
        '''
        obj_value = 0

        # TODO: Index mithilfe von x und grid_size berechnen
        idx = 0,0,0,0,0,0
        obj_value = landscape[idx]
        return obj_value
    print(objective([1,0,5,5,0,0]))
    #[0][0][0][0][0][0]
    print(landscape.shape)

def optimizer_experiment():
    num_layers = 1
    num_qubits = 2
    #num_unitaries = 1 # was 5
    #num_tries = 1 # was 5
    #grid_size = 16
    dimensions = 6 # geht nicht: alles außer 6
    type_of_data = 1 # random data
    deg_of_entanglement = 4 # high, (low = 1)
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
    
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return cost.item()
    
    optimizers = ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP']
    results = {}
    initial_param_values = np.random.uniform(-np.pi, np.pi, size=dimensions)
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
    #single_test_run()
    #run_full_experiment()
    #generate_and_save_testLandscape()
    #run_single_optimizer_experiment()
    optimizer_experiment()

    
    