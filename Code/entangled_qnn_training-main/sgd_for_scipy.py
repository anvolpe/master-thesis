import numpy as np
from scipy.optimize import OptimizeResult
from jax import jacrev
import torch
from scipy.optimize import approx_fprime

"""
    Original by jcmgray: https://gist.github.com/jcmgray/e0ab3458a252114beecb1f4b631e19ab
"""
def sgd(
    fun,
    x0,
    #jac,
    args=(),
    learning_rate=0.001,
    mass=0.9,
    startiter=0,
    maxiter=1000,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of stochastic
    gradient descent with momentum.

    Adapted from ``autograd/misc/optimizers.py``.
    """
    #print(type(x0))
    #x = x0
    #velocity = np.zeros_like(x)
    x = torch.tensor(x0)
    velocity = torch.zeros_like(x)

    for i in range(startiter, startiter + maxiter):
        #g = jac(x)
        #g = approx_fprime(x,fun) # TODO: Besser machen! Braucht sehr lang. Mit torch.tensor irgendwie die Jacobi bestimmen/approx?
        g = torch.autograd.functional.jacobian(fun,x)
        if callback and callback(x):
            break

        velocity = torch.mul(mass,velocity) - torch.mul((1.0 - mass),g)
        x = x + torch.mul(learning_rate,velocity)

    i += 1
    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)


def rmsprop(
    fun,
    x0,
    #jac,
    args=(),
    learning_rate=0.1,
    gamma=0.9,
    eps=1e-8,
    startiter=0,
    maxiter=1000,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of root mean
    squared prop: See Adagrad paper for details.

    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = torch.tensor(x0)
    #velocity = torch.zeros_like(x)
    avg_sq_grad = torch.ones_like(x)

    for i in range(startiter, startiter + maxiter):
        g = torch.autograd.functional.jacobian(fun,x)

        if callback and callback(x):
            break

        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x = x - learning_rate * g / (np.sqrt(avg_sq_grad) + eps)

    i += 1
    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)


def adam(
    fun,
    x0,
    #jac,
    args=(),
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    startiter=0,
    maxiter=1000,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of ADAM -
    [http://arxiv.org/pdf/1412.6980.pdf].

    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = torch.tensor(x0)
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)

    for i in range(startiter, startiter + maxiter):
        g = torch.autograd.functional.jacobian(fun,x)

        if callback and callback(x):
            break

        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1**(i + 1))  # bias correction.
        vhat = v / (1 - beta2**(i + 1))
        x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

    i += 1
    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)