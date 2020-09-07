
import numpy as np
from numpy import sin, pi
from ultranest import ReactiveNestedSampler

def sine_model(t, B, A, P, t0):
    return A * sin((t / P + t0) * 2 * pi) + B


np.random.seed(42)

n_data = 20

# time of observations
t = np.random.uniform(0, 5, size=n_data)

# measurement values
yerr = 1.0
y = np.random.normal(sine_model(t, A=4.2, P=3, t0=0, B=1.0), yerr)

parameters = ['B', 'A', 'P', '$t_0$']

def prior_transform(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales

    params = cube.copy()
    # let background level go from -10 to +10
    params[0] = cube[0] * 20 - 10
    # let amplitude go from 0.01 to 100
    params[1] = 10**(cube[1] * 4 - 2)
    # let period go from 0.3 to 30
    params[2] = 10**(cube[2] * 2)
    # let time go from 0 to 1
    params[3] = cube[3]
    return params



import scipy.stats

def log_likelihood(params):
    # unpack the current parameters:
    B, A, P, t0 = params

    # compute for each x point, where it should lie in y
    y_model = sine_model(t, A=A, B=B, P=P, t0=t0)
    # compute likelihood
    loglike = -0.5 * (((y_model - y) / yerr)**2).sum()

    return loglike



sampler = ReactiveNestedSampler(parameters, log_likelihood, prior_transform,wrapped_params=[False, False, False, True],)

result = sampler.run(min_num_live_points=400, dKL=np.inf, min_ess=100)


sampler.print_results()

