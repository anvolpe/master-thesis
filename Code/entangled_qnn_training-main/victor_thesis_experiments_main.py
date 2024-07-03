import gc
import time
from datetime import datetime
from qnns.cuda_qnn import CudaPennylane

from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import os

def process_sc_metrics(SC):
    """calculates the mean, standard deviation, absolute mean and absolute standard deviations, 
       as well as the positive and negative percentage of the scalar curvature values in the input array

    Args:
        SC (array): the scalar curvature values for each point of a sampled landscape, 
        each entry of the n-dimensional SC array corresponds to the value of the landscape at the same position in the array

    Returns:
        list: a list of all metrics that depend on the scalar curvature array
    """
    sc = np.array(SC).reshape(-1)
    sc_avg = np.mean(sc)
    sc_std = np.std(sc)
    sc_pos = (1.0 * np.sum(sc >= 0)) / (1.0 * len(sc))
    sc_neg = (1.0 * np.sum(sc < 0)) / (1.0 * len(sc))
    sc_abs = np.abs(sc)
    sc_avg_abs = np.mean(sc_abs)
    sc_std_abs = np.std(sc_abs)
    return [sc_avg,sc_std,sc_pos,sc_neg,sc_abs,sc_avg_abs,sc_std_abs]

def process_and_store_metrics(metrics, length, conf_id, experiment_id):
    """calculates, processes and stores the metrics of given landscapes into a txt file for later evaluation
       supposed to get five landscapes, corresponding to 5 different runs for the same configuration
       and unitaries but with different qubit data points
       beings by calculating the metrics for each run individually and then calculates the average and stdev for all runs together

    Args:
        landscapes (array): an array of n dimensional loss landscapes, one landscape for each run with this config
        conf_id (int): the id of the configuration used for these runs
        experiment_id (string): a string identifier for the file system to identify which experiment results and configs belong together
            contains mostly time and dimension/grid size info
    """
    os.makedirs(f"experimental_results/results/runs_{experiment_id}", exist_ok=True)
    file = open(
        f"experimental_results/results/runs_{experiment_id}/conf_{conf_id}.txt", "w"
    )
    file.write(f"conf_id={conf_id}\n---\n")
    file.close()
    TV_arr = metrics[0]
    FD_arr = metrics[1]
    IGSD_arr = metrics[2]
    SC_metrics = metrics[3]

    # calculate and store individual sub-metric (avg, std,..)
    file = open(
        f"experimental_results/results/runs_{experiment_id}/conf_{conf_id}.txt", "a"
    )
    for idx in range(length):
        file.write(f"run_{idx}\n")
        file.write(f"TV={TV_arr[idx]}\n")
        file.write(f"FD={FD_arr[idx]}\n")
        igsd_string = (
            np.array2string(IGSD_arr[idx], separator=",")
            .replace("\n", "")
            .replace(" ", "")
        )
        file.write(f"IGSD={igsd_string}\n")
        # calculate SC sub-metrics
        # flatten SC
        sc_metric = SC_metrics[idx]
        sc_avg = sc_metric[0]
        sc_std = sc_metric[1]
        sc_pos = sc_metric[2]
        sc_neg = sc_metric[3]
        sc_abs = sc_metric[4]
        sc_avg_abs = sc_metric[5]
        sc_std_abs = sc_metric[6]
        file.write(f"SC_pos={sc_pos}\n")
        file.write(f"SC_neg={sc_neg}\n")
        file.write(f"SC_avg={sc_avg}\n")
        file.write(f"SC_std={sc_std}\n")
        file.write(f"SC_avg_abs={sc_avg_abs}\n")
        file.write(f"SC_std_abs={sc_std_abs}\n---\n")
    file.write("combined\n")
    file.close()


def store_configs_to_file(unitaries, configurations, experiment_id):
    """stores all configurations into a txt file,
    sorted by their config id to later relate the config to the results
    config: [type_of_data][num_data_points][deg_of_entanglement][id_unitary][id_try]

    Args:
        unitaries (array of unitary): all unitaries used for the experiments
        configurations (array): arrays of the qubit data point-batches used for the unitary/config combinations
        experiment_id (string): a string identifier for the file system to identify which experiment results and configs belong together
            contains mostly time and dimension/grid size info
    """
    data_type_labels = ["random", "orthogonal", "non_lin_ind", "var_s_rank"]
    conf_id = 0
    file = open(f"experimental_results/configs/configurations_{experiment_id}.txt", "w")
    file.write("")
    file.close()
    for data_type in range(len(configurations)):
        for num_data_points in range(len(configurations[data_type])):
            for s_rank in range(len(configurations[data_type][num_data_points])):
                for unitary_id in range(
                    len(configurations[data_type][num_data_points][s_rank])
                ):
                    out_string = f"conf_id={conf_id}\n"
                    out_string += f"data_type={data_type_labels[data_type]}\n"
                    out_string += f"num_data_points={num_data_points+1}\n"
                    out_string += f"s_rank={s_rank+1}\n"
                    unitary_string = (
                        np.array2string(unitaries[unitary_id].numpy(), separator=",")
                        .replace("\n", "")
                        .replace(" ", "")
                    )
                    out_string += f"unitary={unitary_string}\n"
                    for data_set_id in range(
                        len(
                            configurations[data_type][num_data_points][s_rank][
                                unitary_id
                            ]
                        )
                    ):
                        data_batch = configurations[data_type][num_data_points][s_rank][
                            unitary_id
                        ][data_set_id]
                        data_batch_string = (
                            np.array2string(data_batch.numpy(), separator=",")
                            .replace("\n", "")
                            .replace(" ", "")
                        )
                        out_string += f"data_batch_{data_set_id}={data_batch_string}\n"
                    out_string += "---\n"
                    file = open(
                        f"experimental_results/configs/configurations_{experiment_id}.txt",
                        "a",
                    )
                    file.write(out_string)
                    file.close()
                    conf_id += 1


def generate_data_points(type_of_data, schmidt_rank, num_data_points, U, num_qubits):
    """generates data points given a configuration consisting of the type of data point, 
    the schmidt rank (level of entanglement) 
    and the number of data points, as well as a unitary for reshaping purposes

    Args:
        type_of_data (int): describes what type of data to use (1=random, 2=orthogonal, 3=linearly dependent in H_x, 4= variable schmidt rank)
        schmidt_rank (int): what level of entanglement should the data have
        num_data_points (int): how many data points you want
        U (unitary): a unitary for reshaping of the data points
        num_qubits (int): the amount of wires/x_qubits for the chosen ansatz

    Returns:
        tensor: a tensor of data points that can be used for the experiments
    """

    raw_input = 0
    x_qubits = num_qubits
    r_qubits = x_qubits
    if type_of_data == 1:
        raw_input = torch.from_numpy(
            np.array(
                uniform_random_data(schmidt_rank, num_data_points, x_qubits, r_qubits)
            )
        )
    elif type_of_data == 2:
        raw_input = torch.from_numpy(
            np.array(
                uniformly_sample_orthogonal_points(
                    schmidt_rank, num_data_points, x_qubits, r_qubits
                )
            )
        )
    elif type_of_data == 3:
        raw_input = torch.from_numpy(
            np.array(
                sample_non_lihx_points(
                    schmidt_rank, num_data_points, x_qubits, r_qubits
                )
            )
        )
    elif type_of_data == 4:
        raw_input = torch.from_numpy(
            np.array(
                uniform_random_data_average_evenly(
                    schmidt_rank, num_data_points, x_qubits, r_qubits
                )
            )
        )
    return raw_input.reshape(
        (raw_input.shape[0], int(raw_input.shape[1] / U.shape[0]), U.shape[0])
    ).permute(0, 2, 1)

# alle Optimierer für alle bestimmte Spezifikation + objective function laufen lassen + alles speichern (???)
def run_optimizer_experiments(landscape, grid_size, conf_id, experiment_id):
    # TODO: def objective
    # TODO: Schleife über alle Optimierer & deren Spezifikationen
    # TODO: 

    def objective(x):
        return x


# TODO: wenn ich's richtig verstanden hab, müssen wir nur hier was ändern
# eventuell direkt in dieser Methode oder (besser?) eine neue Methode schreiben, 
# die dann die generierte Loss Landscape minimiert & alles in der richtigen Datei speichert
# In dieser Methode: alle optimierer & Spezifikations Kombis nacheinander durchmachen 
# --> Nur eine handvoll Loss Landscape untersuchen?
def run_single_experiment_batch(
    grid_size, dimensions, data_batch, U, qnn, conf_id, experiment_id
):
    """this runs the experiment (calculating landscapes and metrics, and storing the metrics) 
    on one config/unitary combination on one thread/core.
    Right now this will calculate 5 landscapes for one combination, 
    one for each set of datapoints provided in the data batch

    Args:
        grid_size (int): the wanted resolution of the landscapes
        dimensions (int): how many dimensions will the landscape have
        data_batch (array): a batch of data points, one for each run on this config/unitary combination
        U (unitary): the unitary the qnn will be compared against
        conf_id (int): the id of the configuration used for the data points
        experiment_id (string): a string identifier for the file system to identify which experiment results and configs belong together
            contains mostly time and dimension/grid size info
    """
    TV_arr = []
    FD_arr = []
    IGSD_arr = []
    SC_metrics = []
    for data_set in data_batch:
        landscape = generate_loss_landscape(grid_size, dimensions, data_set, U, qnn)
        TV_arr.append(calc_total_variation(landscape))
        FD_arr.append(calc_fourier_density(landscape))
        IGSD_arr.append(calc_IGSD(landscape))
        SC = calc_scalar_curvature(landscape)
        SC_metrics.append(process_sc_metrics(SC))
        del SC
        # del landscape # weglassen?
        # TODO: def objective basierend auf landscape
        # TODO: optimize.minimize(...) Aufruf für einen Optimierer + Options (Spezifikationen)
        # TODO: Zeit messen + Zeit speichern

        gc.collect() # garbage collector
        
    metrics = []
    metrics.append(TV_arr)
    metrics.append(FD_arr)
    metrics.append(IGSD_arr)
    metrics.append(SC_metrics)
    process_and_store_metrics(metrics, len(data_batch), conf_id, experiment_id)
    # TODO: speichern: Optimierer, Zeiten, Spezifikation
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"[{now}] Finished run: {conf_id}")


def run_full_experiment():
    """this function contains the main experiment
    here it will calculate different configurations for the data points used as qubits 
    (by data type, schmidt rank and amount of data points)
    then it will run these configurations of five different unitaries per config
    and with five sets of data points (generated according to the config)
    where a unitary/datapoint config will run all 5 sets of datapoints (data batch) on one core
    with many of these combinations running in parallel
    """
    num_layers = 1
    num_qubits = 2
    num_unitaries = 5
    num_tries = 5
    grid_size = 16
    dimensions = 6
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

    os.makedirs("experimental_results/configs", exist_ok=True)
    os.makedirs("experimental_results/results", exist_ok=True)
    # generate a U3 ansatz containing 2 layers -> 6 params
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") # TODO: läuft nicht wegen qml.matrix (wire_order is required)

    unitaries = []
    # [type_of_data][num_data_points][deg_of_entanglement][id_unitary][id_try]
    configurations = []
    for _ in range(num_unitaries):
        # generate a random unitary with num_qubits qubits (why are they the same?)
        unitaries.append(
            torch.tensor(
                np.array(random_unitary_matrix(num_qubits)),
                dtype=torch.complex128,
                device="cpu",
            )
        )

    start = time.time()
    # generate configurations (5 datapoint sets = 5 runs per config)
    conf_id = 0
    # cpu_count()
    with ProcessPoolExecutor(cpu_count()) as exe:
        # iterate over  type of training data: 1=random, 2=orthogonal, 3=linearly dependent in H_x, 4= variable schmidt rank
        # TODO: Wie können wir die richtigen Trainingsdaten auswählen (es sind bestimmt nicht alle nötig)? nur über Schmidt Rang? 
        for type_of_data in range(1, 5, 1):
            num_data_points_row = []
            # iterate over training data size 1 to 4
            for num_data_points in range(1, 5, 1):
                deg_of_entanglement_row = []
                # iterate over degree of entanglement 1 to 4 
                # hier einfach nur die trainingsdaten mit hohem degree of entanglement auswählen?
                for deg_of_entanglement in range(1, 5, 1):
                    # iterate over unitaries
                    unitary_row = []
                    for unitary in unitaries:
                        data_batch_for_unitary = []
                        # iterate over number of tries/runs
                        for _ in range(1, num_tries + 1, 1):
                            # generate array of training data configurations [type_of_data][num_data_points][deg_of_entanglement][id_unitary][id_try]
                            data_points = generate_data_points(
                                type_of_data,
                                deg_of_entanglement,
                                num_data_points,
                                unitary, num_qubits
                            )
                            data_batch_for_unitary.append(data_points)
                        # run this per configuration unitary (5 sets of data -> take average and stdv...)
                        exe.submit(
                            run_single_experiment_batch, 
                            grid_size,
                            dimensions,
                            data_batch_for_unitary,
                            unitary,
                            qnn,
                            conf_id,
                            exp_id,
                        )
                        # run_single_experiment_batch(grid_size, dimensions, data_batch_for_unitary, unitary, qnn, conf_id, exp_id)
                        conf_id += 1
                        unitary_row.append(data_batch_for_unitary)
                    deg_of_entanglement_row.append(unitary_row)
                num_data_points_row.append(deg_of_entanglement_row)
            configurations.append(num_data_points_row)

    store_configs_to_file(unitaries, configurations, exp_id)
    end = time.time()
    print(f"total runtime: {np.round(end-start,2)}s")


if __name__ == "__main__":
    # one thread per core
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy("file_system")
    run_full_experiment()
