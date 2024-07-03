import os
import json
import h5py
import pyccl
import numpy
import argparse

def main(z1, z2, bin_size, cosmo_ccl):
    """
    Calculate the redshift bins for a given range of redshifts and bin size.
    
    Parameters:
        z1 (float): The lower redshift limit.
        z2 (float): The upper redshift limit.
        bin_size (int): The number of bins to divide the redshift range into.
        cosmo_ccl (pyccl.Cosmology): The CCL cosmology object.
    
    Returns:
        numpy.ndarray: An array of redshift values representing the bin edges.
    
    """
    chi1 = pyccl.background.comoving_radial_distance(cosmo=cosmo_ccl, a=1 / (1+z1))
    chi2 = pyccl.background.comoving_radial_distance(cosmo=cosmo_ccl, a=1 / (1+z2))
    
    chi_bin = numpy.linspace(chi1, chi2, bin_size + 1)
    z_bin = 1 / pyccl.background.scale_factor_of_chi(cosmo=cosmo_ccl, chi=chi_bin) - 1
    
    return z_bin

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Informer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    
    PATH = PARSE.parse_args().path
    DATA_PATH = os.path.join(PATH, 'DATA')
    
    with open(DATA_PATH + '/BIN/COSMO.json', 'r') as cosmo:
        COSMO = json.load(cosmo)
    
    COSMO_CCL = pyccl.Cosmology(
        h = COSMO['H'],
        w0 = COSMO['W0'],
        wa = COSMO['WA'], 
        n_s = COSMO['NS'], 
        A_s = COSMO['AS'],
        m_nu = COSMO['MNU'],  
        Neff = COSMO['NEFF'],
        T_CMB = COSMO['TCMB'], 
        Omega_k = COSMO['OMEGAK'], 
        Omega_c = COSMO['OMEGAC'], 
        Omega_b = COSMO['OMEGAB'], 
        matter_power_spectrum = 'halofit',  
        transfer_function = 'boltzmann_camb', 
        extra_parameters = {'camb': {'kmax': 100, 'lmax': 10000, 'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.8}}
    )
    
    z1_lens = 0.0
    z2_lens = 2.0
    lens_size = 5
    bin_lens = main(z1_lens, z2_lens, lens_size, COSMO_CCL)
    
    z1_source = 0.0
    z2_source = 3.0
    source_size = 5
    bin_source = main(z1_source, z2_source, source_size, COSMO_CCL)
    
    with h5py.File(DATA_PATH + '/BIN/BIN.hdf5', 'w') as file:
        file.create_dataset('lens', data=bin_lens)
        file.create_dataset('source', data=bin_source)