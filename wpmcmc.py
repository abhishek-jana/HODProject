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

'''
from scipy.optimize import minimize

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = x_truth + 0.1 * np.random.randn(5)
soln = minimize(nll, initial, args=(wp_data, wp_cov))
M_cut_ml,M1_ml,sigma_ml, kappa_ml, alpha_ml = soln.x

print (soln.x)
'''
import emcee
ndim = 5
nwalkers = 50
#init = np.array([13.13832939, 14.04617357,  1.04476888,  1.28230299,  0.87658467])
init = np.array([13.09006542, 14.05997246,  0.98009625,  1.1310932,   0.90094299])
pos = init + 1e-4 * np.random.rand(nwalkers, ndim)




with Pool() as pool:
    filename = "wpmcmctest.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    print ("Running burn-in...")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(wp_data, wp_cov),pool=pool,backend=backend)
    #pos, _, _ = sampler.run_mcmc(pos, 200, progress = True, store = False)
    #sampler.reset()
    print("Running production...")
    sampler.run_mcmc(pos, 5000,store=True, progress=True)
    
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

print (np.median(sampler.flatchain, axis=0))
