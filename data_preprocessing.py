import numpy as np
import os

path_vpf = '/mnt/data4/Abhishek/VPF/random/'
path_cic = '/mnt/data4/Abhishek/CIC/random/'
path_cia = '/mnt/data4/Abhishek/CIA/random/'


def extractIndex(filename):
        return int(filename.split('_')[-1][:4])

f = np.loadtxt('/home/ajana/github/HODProject/parameters.txt')

full_data = []

for filename in os.listdir(path_cic):
    if filename.startswith('cic_galaxies'):
        pos = extractIndex(filename)
        cic = np.load(os.path.join(path_cic,filename))
        hist_cic,_ = np.histogram(cic,bins=15)
        hist_cic = hist_cic/np.sum(hist_cic)
        del cic
        hist_cic[np.where(hist_cic==0)] = 1e-10
        cia = np.load(os.path.join(path_cia,'cia_'+filename[4:]))
        hist_cia,_ = np.histogram(cia,bins=15)
        hist_cia = hist_cia/np.sum(hist_cia)
        del cia
        hist_cia[np.where(hist_cia==0)] = 1e-10
        _,vpf = np.load(os.path.join(path_vpf,'vpf_'+filename[4:]))
        _temp = np.concatenate((vpf,np.log(hist_cic),np.log(hist_cia),f[pos]))
        del hist_cic
        del hist_cia
        del vpf
        full_data.append(_temp)
        del _temp

full_data = np.vstack((full_data))

np.save('machine_learning_data.npy',full_data)

print (f'Shape of data : {full_data.shape}') 
