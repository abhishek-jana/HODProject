import emcee
import numpy as np
import os
import joblib
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")


np.random.seed(42)


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

wpz_cov = np.cov(wp_z)
wpx_cov = np.cov(wp_x)
wpy_cov = np.cov(wp_y)
wp_cov = (wpz_cov + wpx_cov + wpy_cov)/3

filename = 'wp_model.sav'
# load the model from disk
model = joblib.load(filename)
wp_data = np.load('/mnt/data4/Abhishek/WP/wp_galaxies_0000.npy')[0]
x_truth = np.array([13.088658, 14.060000, 0.980000, 1.130000, 0.900000])

def wp_model(par):
    result = model.predict(par.reshape(1,-1))
    return result[0]

def log_likelihood(par , data, cov):
    M_cut,M1 ,sigma, kappa, alpha = par
    diff = data - wp_model(par)
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

# Initialize the walkers
#coords = np.random.randn(32, 5)
nwalkers, ndim = 200,5
    



with Pool() as pool:
    filename = "wpmcmctest.h5"
    backend = emcee.backends.HDFBackend(filename)
    print("Initial size: {0}".format(backend.iteration))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(wp_data, wp_cov),pool=pool,backend=backend)
    sampler.run_mcmc(None, 5000, progress=True)
    print("Final size: {0}".format(backend.iteration))
    
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

print (np.median(sampler.flatchain, axis=0))
