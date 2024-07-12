import os
import json
import time
import h5py
import numpy
import pyccl
import argparse
import multiprocessing


def bin(z1, z2, bin_size, cosmo_ccl):
    
    chi1 = pyccl.background.comoving_radial_distance(cosmo=cosmo_ccl, a=1 / (1 + z1))
    chi2 = pyccl.background.comoving_radial_distance(cosmo=cosmo_ccl, a=1 / (1 + z2))
    
    chi_bin = numpy.linspace(chi1, chi2, bin_size + 1)
    z_bin = 1 / pyccl.background.scale_factor_of_chi(cosmo=cosmo_ccl, chi=chi_bin) - 1
    return z_bin


def main(path, index):
    start = time.time()
    
    # Data
    data_path = os.path.join(path, 'DATA/')
    
    # Redshift
    bin_size = 5
    z1_lens = 0.0
    z2_lens = 2.0
    
    # cosmology
    with open(data_path + '/COSMO/COSMO.json', 'r') as file:
        cosmo = json.load(file)
    
    cosmo_ccl = pyccl.cosmology.Cosmology(
        h=cosmo['H'],
        w0=cosmo['W0'],
        wa=cosmo['WA'],
        n_s=cosmo['NS'],
        m_nu=cosmo['MNU'],
        Neff=cosmo['NEFF'],
        T_CMB=cosmo['TCMB'],
        mass_split='single',
        sigma8=cosmo['SIGMA8'],
        Omega_k=cosmo['OMEGAK'],
        Omega_c=cosmo['OMEGAC'],
        Omega_b=cosmo['OMEGAB'],
        mg_parametrization=None,
        matter_power_spectrum='halofit',
        transfer_function='boltzmann_camb',
        extra_parameters={
            'camb': {'kmax': 10000, 'lmax': 10000, 'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.8}}
    )
    
    # Select
    bin_lens = bin(z1_lens, z2_lens, bin_size, cosmo_ccl)
    
    # Save
    os.makedirs(os.path.join(data_path, 'SELECT/LENS/LENS{}'.format(index)), exist_ok=True)
    with h5py.File(os.path.join(data_path, 'SELECT/LENS/LENS{}/BIN.hdf5'.format(index)), 'w') as file:
        file.create_dataset('bin', data=bin_lens)
    
    # Return
    end = time.time()
    print('Index {}, Time: {:.2f} minutes'.format(index, (end - start) / 60))
    return bin_lens
    
if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Informer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the processes')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets')
    
    PATH = PARSE.parse_args().path
    NUMBER = PARSE.parse_args().number
    LENGTH = PARSE.parse_args().length
    
    # Multiprocessing
    SIZE = LENGTH // NUMBER
    for CHUNK in range(SIZE):
        print('CHUNK: {}'.format(CHUNK + 1))
        with multiprocessing.Pool(processes=NUMBER) as POOL:
            POOL.starmap(main, [(PATH, INDEX) for INDEX in range(CHUNK * NUMBER + 1, (CHUNK + 1) * NUMBER + 1)])