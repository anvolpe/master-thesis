import torch
import numpy as np
from classic_training import cost_func
from data import *
from generate_experiments import get_qnn
import numpy as np
from utils import *
from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *


def generate_random_datapoints(numb_points, s_rank, U):
    """generates random sample datapoints for a qnn

    Args:
        numb_points (int): the number of datapoints you want to generate
        s_rank (int): the schmidt rank ("level of entanglement") of the data points (with the actual qbits and the qbits for the reference system) 
        U (unitary): unitary
    Returns:
        tensor: data points used as qubit inputs for qnns
    """
    schmidt_rank = s_rank
    num_points = numb_points
    x_qbits = 1
    r_qbits = s_rank - x_qbits
    inputs = torch.from_numpy(
        np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits))
    )
    inputs = inputs.reshape(
        (inputs.shape[0], int(inputs.shape[1] / U.shape[0]), U.shape[0])
    ).permute(0, 2, 1)
    return inputs


def get_zero_one_datapoints():
    """generates zero and one vectors as data points

    Returns:
        tensor: a tensor containing the zero and the one vector as data points
    """
    zero_state = np.array([[1], [0]], dtype=complex)
    one_state = np.array([[0], [1]], dtype=complex)
    tensor = torch.tensor(np.array([zero_state, one_state]))
    return tensor

# gen n-d loss landscape
def generate_loss_landscape(grid_size, dimensions, inputs, U, qnn):
    """generates an n-dimensional loss landscape

    Args:
        grid_size (int): the sampling resolution in every dimension(=direction)
        dimensions (int): how many dimensions should be sampled
        inputs (tensor): a tensor of data points for which the qnn will be evaluated
        U (unitary): the unitary which the qnn is trying to emulate

    Returns:
        array: n dimensional loss landscape
    """
    x = inputs
    expected_output = torch.matmul(U, x)
    y_true = expected_output.conj()
    # calculate the parameter values for the grid size, evenly spread from 0 to 2 pi
    param_vals = []
    lanscape_limit = 2 * math.pi
    step_size = lanscape_limit / grid_size
    #step_size = lanscape_limit / (grid_size-1) # <- more evenly spread samples
    for step in range(grid_size):
        param_vals.append(step*step_size) # TODO: was ist param_vals??????
    # generate landscape
    landscape_shape = []
    # 5, 9 [9][9][9][9][9]
    for _ in range(dimensions):
        landscape_shape.append(grid_size)
    landscape_shape = tuple(landscape_shape)
    landscape = np.empty(landscape_shape)
    # for every point
    print("param_vals", param_vals)
    for idx, _ in np.ndenumerate(landscape):  
        param_list = []      
        # generate param array
        #print("idx", idx)
        for dimension in idx: # idx = [a,b,c,d,e,f] wobei alle zwischen 0 und 15 sind --> index eines Gitterpunkts, welcher Parameter für qnn is
            param_list.append(param_vals[dimension]) 
        # calculate cost function
        param_list = np.asarray(param_list) # Gitterpunkt x für objective(x)
        #print("param_vals", param_vals)
        #print("param_list", param_list)
        qnn.params = torch.tensor(param_list, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape)
        #print("qnn.params",qnn.params)
        cost = cost_func(inputs, y_true, qnn, device="cpu") 
        landscape[idx]=cost.item()
    return landscape

# gen 2D loss landscape
def generate_2d_loss_landscape(grid_size, inputs, U, qnn):
    """generates a 2D loss landscape

    Args:
        grid_size (int): the sampling resolution for the loss landscape
        inputs (tensor): tensor representation of the data points given to the qnn
        U (unitary): unitary which the qnn tries to emulate

    Returns:
        array: a 2D loss landscape
    """
    landscape = []
    lanscape_limit = 2 * math.pi
    step_size = lanscape_limit / grid_size
    #step_size = lanscape_limit / (grid_size-1) # <- more evenly spread samples
    x = inputs
    expected_output = torch.matmul(U, x)
    y_true = expected_output.conj()
    for i in range(grid_size):
        # start at 2pi so y axis label still fits (upwards scaling instead of downards)
        #arg_1 = lanscape_limit - i * step_size
        arg_1 = i * step_size
        row = []
        for j in range(grid_size):
            # start at 0 because x axis label direction is correct
            arg_2 = j * step_size
            qnn.params = torch.tensor(
                [[[arg_1, arg_2]]], dtype=torch.float64, requires_grad=True
            )
            cost = cost_func(inputs, y_true, qnn, device="cpu")
            row.append(cost.item())
        landscape.append(row)
    return landscape


# gen 3D loss landscape for U3
def generate_3D_loss_landscape_with_labels(grid_size, inputs, U):
    """generates a 3D loss landscape using the PennyLane (U3) ansatz
    also returns the labels for if you want to plot this landscape

    Args:
        grid_size (int): sets the resolution of the resulting loss landscape
        inputs (tensor): a tensor of data points for which the qnn will be evaluated
        U (unitary): the unitary which the qnn is trying to emulate

    Returns:
        array: a 3D loss landscape
        array: the labels for the landscape
    """
    qnn = get_qnn("CudaPennylane", list(range(1)), 1, device="cpu")
    landscape = []
    x_array = []
    y_array = []
    z_array = []
    points = []
    lanscape_limit = 2 * math.pi
    step_size = lanscape_limit / grid_size
    x = inputs
    expected_output = torch.matmul(U, x)
    y_true = expected_output.conj()
    for i in range(grid_size):
        # start at 2pi so y axis label still fits (upwards scaling instead of downards)
        arg_1 = i * step_size
        for j in range(grid_size):
            # start at 0 because x axis label direction is correct
            arg_2 = lanscape_limit - j * step_size
            for k in range(grid_size):
                # maybe change direction?
                arg_3 = k * step_size
                qnn.params = torch.tensor(
                    [[[arg_1, arg_2, arg_3]]], dtype=torch.float64, requires_grad=True
                )
                cost = cost_func(inputs, y_true, qnn, device="cpu")
                landscape.append(cost.item())
                x_array.append(arg_3)
                y_array.append(arg_2)
                z_array.append(arg_1)
    points.append(x_array)
    points.append(y_array)
    points.append(z_array)
    return landscape, points
 

def generate_3D_loss_landscape(grid_size, inputs, U):
    """generates a 3D loss landscape using the PennyLane (U3) ansatz

    Args:
        grid_size (int): sets the resolution of the resulting loss landscape
        inputs (tensor): a tensor of data points for which the qnn will be evaluated
        U (unitary): the unitary which the qnn is trying to emulate

    Returns:
        array: a 3D loss landscape
    """
    qnn = get_qnn("CudaPennylane", list(range(1)), 1, device="cpu")
    landscape = []
    lanscape_limit = 2 * math.pi
    step_size = lanscape_limit / grid_size
    x = inputs
    expected_output = torch.matmul(U, x)
    y_true = expected_output.conj()
    for i in range(grid_size):
        row_x= []
        # start at 2pi so y axis label still fits (upwards scaling instead of downards)
        arg_1 = i * step_size
        for j in range(grid_size):
            row_y = []
            # start at 0 because x axis label direction is correct
            arg_2 = lanscape_limit - j * step_size
            for k in range(grid_size):
                # maybe change direction?
                arg_3 = k * step_size
                qnn.params = torch.tensor(
                    [[[arg_1, arg_2, arg_3]]], dtype=torch.float64, requires_grad=True
                )
                cost = cost_func(inputs, y_true, qnn, device="cpu")
                row_y.append(cost.item())
            row_x.append(row_y)
        landscape.append(row_x)
    return landscape
