import numpy as np
import joblib
import os
from multiprocessing import Pool
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

wp = (wp_x + wp_y + wp_z)/3

#wpz_cov = np.cov(wp_z)
#wpx_cov = np.cov(wp_x)
#wpy_cov = np.cov(wp_y)
#wp_cov = (wpz_cov + wpx_cov + wpy_cov)/3

vpf_path = '/mnt/data4/Abhishek/fidmock/vpf'

vpf_filenames = ['vpf_subsample_'+f'{files:04d}'+'.npy' for files in range(100)]
vpf = []
for files in vpf_filenames:
    sample = np.load(os.path.join(vpf_path,files))
    vpf.append(sample[1])
vpf_z = np.vstack(vpf).T  
vpf_filenames = ['vpf_subsample_'+f'{files:04d}'+'.npy' for files in range(100,200)]
vpf = []
for files in vpf_filenames:
    sample = np.load(os.path.join(vpf_path,files))
    vpf.append(sample[1])
vpf_x = np.vstack(vpf).T 
vpf_filenames = ['vpf_subsample_'+f'{files:04d}'+'.npy' for files in range(200,300)]
vpf = []
for files in vpf_filenames:
    sample = np.load(os.path.join(vpf_path,files))
    vpf.append(sample[1])
vpf_y = np.vstack(vpf).T 

vpf = (vpf_x + vpf_y + vpf_z)/3

#vpfz_cov = np.cov(vpf_z)
#vpfx_cov = np.cov(vpf_x)
#vpfy_cov = np.cov(vpf_y)
#vpf_cov = (vpfz_cov + vpfx_cov + vpfy_cov)/3

total = np.vstack((wp,vpf))
total_cov = 100*np.cov(total,bias = True)

wp_filename = 'wp_model.sav'
# load the model from disk
wpmodel = joblib.load(wp_filename)
wp_data = np.load('/mnt/data4/Abhishek/WP/wp_galaxies_0000.npy')[0]



vpf_filename = 'vpf_model.sav'
# load the model from disk
vpfmodel = joblib.load(vpf_filename)
vpf_data = np.load('/mnt/data4/Abhishek/VPF/random/vpf_galaxies_0000.npy')[1]

total_data = np.concatenate((wp_data,vpf_data))

x_truth = np.array([13.088658, 14.060000, 0.980000, 1.130000, 0.900000])

def wp_model(par):
    result = wpmodel.predict(par.reshape(1,-1))
    return result[0]

def vpf_model(par):
    result = vpfmodel.predict(par.reshape(1,-1))
    return result[0]

def log_likelihood(par , data, cov):
    M_cut,M1 ,sigma, kappa, alpha = par
    diff = data - np.concatenate((wp_model(par),vpf_model(par)))
    return  -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

def log_prior(par):
    M_cut,M1 ,sigma, kappa, alpha = par
    if 12.0 < M_cut < 15.0 and 12.0 < M1 < 15.0 and 1e-8 < sigma < 2.0 and 1e-8 < kappa < 2.0 and 1e-8 < alpha < 2.0:
        return 0.0
    return -np.inf

def log_probability(par, data, cov):
    lp = log_prior(par)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(par, data, cov)

#print (total_data)
#print(wp_model(x_truth))

'''
from scipy.optimize import minimize

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = x_truth + 0.1 * np.random.randn(5)
soln = minimize(nll, initial, args=(total_data, total_cov))
M_cut_ml,M1_ml,sigma_ml, kappa_ml, alpha_ml = soln.x

print (soln.x)
'''

import emcee
ndim = 5
nwalkers = 64

init = np.array([13.08899044, 14.05971487,  0.97996562,  1.12750638,  0.90120769])
pos = init + 1e-4 * np.random.rand(nwalkers, ndim)
'''
with Pool() as pool:
    filename = "wpvpfmcmctest.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    #print ("Running burn-in...")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(total_data, total_cov),pool=pool,backend=backend)
    #pos, _, _ = sampler.run_mcmc(pos, 100, progress = True, store = False)
    #sampler.reset()
    print("Running production...")
    sampler.run_mcmc(pos, 5000,store=True, progress=True)
'''    
   
# Resume from saved chain

with Pool() as pool:
    filename = "wpvpfmcmctest.h5"
    backend = emcee.backends.HDFBackend(filename)
    print("Initial size: {0}".format(backend.iteration))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(total_data, total_cov),pool=pool,backend=backend)
    print("Running production...")
    sampler.run_mcmc(None, 10000,store=True, progress=True)
    print("Final size: {0}".format(backend.iteration))

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

print (np.median(sampler.flatchain, axis=0))

