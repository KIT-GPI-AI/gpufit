import os


# COMMANDS TO RECOMPILE C-LIB AND PYTHON BINDINGS
#==================================================================

# Make sure you have seispy, cuda and cudnn loaded (ml seispy cuda cudnn)

# cd /scratch/gpi/seis/HEX/GpuFit/Gpufit-build/
# cmake -DCMAKE_BUILD_TYPE=RELEASE ../Gpufit
# make
# cd /scratch/gpi/seis/HEX/GpuFit/Gpufit-build/pyGpufit/dist/
# pip uninstall pyGpufit
# pip install --force-reinstall pyGpufit-1.2.0-py2.py3-none-any.whl pyGpufit

# cd ../../

#===================================================================

import pygpufit.gpufit as gf
import numpy as np
from matplotlib import pyplot as plt

# cuda available checks
print("CUDA available: {}".format(gf.cuda_available()))
# if not gf.cuda_available():
#    raise RuntimeError(gf.get_last_error())
print("CUDA versions runtime: {}, driver: {}".format(*gf.get_cuda_version()))
print("Using GPU...")


# DEFINE INPUT PARAMETERS
#==================================================================
# number of fits and fit points
number_fits = 1000000
size_x = 10
number_points = size_x
number_parameters = 5

# Bounds for initial guesses for each parameter sampled ~U[min, max]
min_x, max_x = 40, 60
min_y, max_y = 40, 60
min_z, max_z = 10, 30
min_t0, max_t0 = 5, 15
min_v, max_v = 5, 7.5

tolerance = 0.00001
max_number_iterations = 200

#estimator_id = gf.EstimatorID.LSE
estimator_id = gf.EstimatorID.MLE
model_id = gf.ModelID.HYPERBOLA


# GENERATE INPUT DATA
#==================================================================
# true parameters  (x0, y0, z0, t0, v)
true_parameters = np.array((20, 30, 31, 10, 8), dtype=np.float32)
guess_parameters =np.array((20, 30, 12, 7, 7), dtype=np.float32)

print('true parameters:', true_parameters.astype(list))


# initialize random number generator
np.random.seed(0)

# Generate random guesses for initial parameters
initial_parameters = np.tile(guess_parameters, (number_fits, 1))
# for row_i in range(initial_parameters.shape[0]):
#     initial_parameters[row_i, 0] = np.random.uniform(min_x, max_x)
#     initial_parameters[row_i, 1] = np.random.uniform(min_y, max_y)
#     initial_parameters[row_i, 2] = np.random.uniform(min_z, max_z)
#     initial_parameters[row_i, 3] = np.random.uniform(min_t0, max_t0)
#     initial_parameters[row_i, 4] = np.random.uniform(min_v, max_v)
initial_parameters = initial_parameters.astype(np.float32)

print('Initial parameter guesses')
print(initial_parameters)
print('\n')

# Forward model hyperbola
def hyperbola(X, x0, y0, z0, t0, v):
    r = np.sqrt((X[0, :] - x0) ** 2 + (X[1, :] - y0) ** 2 + (X[2, :] - z0) ** 2)
    tt = t0**2 + (r**2 / v**2)
    return tt


# generate X values (n-fits, n-points, 3)
x = np.random.uniform(-90, 90, size_x)
y = np.random.uniform(-90, 90, size_x)
z = np.random.uniform(0, 1, size_x)

user_info_list = []

for fit in range(number_fits):
    for i in range(z.shape[0]):
        user_info_list.append(x[i])
        user_info_list.append(y[i])
        user_info_list.append(z[i])

user_info = np.array(user_info_list)
user_info = user_info.astype(np.float32)
#user_info.dtype=np.float32
print(user_info)

x_2d = user_info.reshape(number_fits*number_points, 3).T
print(x_2d.astype(list))
print(x_2d.shape, '<--- is it (3, 28)?')

# create y-data
data = hyperbola(x_2d, true_parameters[0],true_parameters[1], true_parameters[2], true_parameters[3], true_parameters[4]) # data = np.array([hyperbola(xi[i].T, *true_parameters) for i in range(number_fits)])

# data = np.arange(data.shape[0])

# add noise
data = data + np.random.normal(0.0,1.0, size=data.shape)
data = data.astype(np.float32)
print('data', data.shape)
print(data)

#print('reshaping data')
#print(data.reshape(7, 4).T.astype(list))


#print('checking shapes')
#print(data.reshape(7, 4).T.shape)
#print(data.reshape(4, 7).shape)

#print('user-info', xi.shape)
#print(user_info.astype(list))

data = data.reshape(number_fits, number_points)
data = data.astype(np.float32)

# FIT
#==================================================================
# run Gpufit
parameters, states, chi_squares, number_iterations, execution_time = gf.fit(
    data,           # <------ PREDICTION VALUES (y)    -- shape(n-fits, n-data-points)
    None,
    model_id,
    initial_parameters,
    tolerance,
    max_number_iterations,
    None,
    estimator_id,
    user_info=user_info,  # <------- DEPENDANT VARIABLES (X) -- shape(3, n-fits, n-data-points).flatten()
)

# print fit results
converged = states == 0
print("*Gpufit*")

# print summary
print("\nmodel ID:        {}".format(model_id))
print("number of fits:  {}".format(number_fits))
print("fit size:        {}".format(size_x))
print("mean chi_square: {:.2f}".format(np.mean(chi_squares[converged])))
print("iterations:      {:.2f}".format(np.mean(number_iterations[converged])))
print("time:            {:.2f} s".format(execution_time))

# get fit states
number_converged = np.sum(converged)
print(
    "\nratio converged         {:6.2f} %".format(number_converged / number_fits * 100)
)
print(
    "ratio max it. exceeded  {:6.2f} %".format(np.sum(states == 1) / number_fits * 100)
)
print(
    "ratio singular hessian  {:6.2f} %".format(np.sum(states == 2) / number_fits * 100)
)
print(
    "ratio neg curvature MLE {:6.2f} %".format(np.sum(states == 3) / number_fits * 100)
)

# mean, std of fitted parameters
converged_parameters = parameters[converged, :]
converged_parameters_mean = np.median(converged_parameters, axis=0)
converged_parameters_std = np.std(converged_parameters, axis=0)
print("\nparameters of line")
for i in range(number_parameters):
    print(
        "p{} true {:6.2f} mean {:6.2f} std {:6.2f}".format(
            i,
            true_parameters[i],
            converged_parameters_mean[i],
            converged_parameters_std[i],
        )
    )
