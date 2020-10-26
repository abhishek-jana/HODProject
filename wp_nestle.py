import numpy as np
import os
import nestle
import joblib
import warnings
warnings.filterwarnings("ignore")

wp_path = '/mnt/data4/Abhishek/fidmock/wp'

wp_filenames = ['wp_subsample_'+f'{files:04d}'+'.npy' for files in range(100)]
wp = []
for files in wp_filenames:
    sample = np.load(os.path.join(wp_path,files))
    wp.append(sample[0])
wp_z = np.vstack(wp).T  
wp_filenames = ['wp_subsample_'+f'{files:04d}'+'.npy' for files in range(100,200)]
wp = []
for files in wp_filenames:
    sample = np.load(os.path.join(wp_path,files))
    wp.append(sample[0])
wp_x = np.vstack(wp).T 
wp_filenames = ['wp_subsample_'+f'{files:04d}'+'.npy' for files in range(200,300)]
wp = []
for files in wp_filenames:
    sample = np.load(os.path.join(wp_path,files))
    wp.append(sample[0])
wp_y = np.vstack(wp).T 


wpz_cov = np.cov(wp_z,bias = True)
wpx_cov = np.cov(wp_x,bias = True)
wpy_cov = np.cov(wp_y,bias = True)
wp_cov = 100*(wpz_cov + wpx_cov + wpy_cov)/3

filename = 'wp_model.sav'
# load the model from disk
model = joblib.load(filename)
wp_data = np.load('/mnt/data4/Abhishek/WP/wp_galaxies_0000.npy')[0]
x_truth = np.array([13.088658, 14.060000, 0.980000, 1.130000, 0.900000])

def wp_model(par):
    par = np.array(par).reshape(1,-1)
    result = model.predict(par)
    return result[0]

parameters = ['M_cut','M1','sigma', 'kappa', 'alpha']


def prior_transform(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales

    params = cube.copy()
    # let slope go from 12.0 to 15.0
    lo = 12.0
    hi = 15.0
    M_cut = cube[0] * (hi - lo) + lo
    # let slope go from 12.0 to 15.0
    lo = 12.0
    hi = 15.0
    M1 = cube[1] * (hi - lo) + lo
    # let slope go from 1e-8 to 2.0
    lo = 1e-8
    hi = 2.0
    sigma = cube[2] * (hi - lo) + lo
    # let slope go from 1e-8 to 2.0
    lo = 1e-8
    hi = 2.0
    kappa = cube[3] * (hi - lo) + lo
    # let slope go from 1e-8 to 2.0
    lo = 1e-8
    hi = 2.0
    alpha = cube[4] * (hi - lo) + lo    
    return (M_cut,M1 ,sigma, kappa, alpha)




def log_likelihood(par):
    M_cut,M1 ,sigma, kappa, alpha = par
    diff = wp_data - wp_model(par)
    return  -0.5 * np.dot(diff, np.linalg.solve(wp_cov, diff))

#print (wp_model(x_truth))

nlive = 1     # number of live points
method = 'multi' # use MutliNest algorithm
ndims = 5        # two parameters
tol = 0.1        # the stopping criterion

res = nestle.sample(log_likelihood, prior_transform, ndims, method=method, npoints=nlive, dlogz=tol)

logZnestle = res.logz                         # value of logZ
infogainnestle = res.h                        # value of the information gain in nats
logZerrnestle = np.sqrt(infogainnestle/nlive) # estimate of the statistcal uncertainty on logZ

print("log(Z) = {} ± {}".format(logZnestle, logZerrnestle))

print(res.summary())



