import os
import json
import time
import numpy
import pyccl
import argparse


def main(folder):
    '''
    Store the fiducial values of intrinsic alignment
    
    Arguments:
        folder (str): The base folder of the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    info_folder = os.path.join(folder, 'INFO/')
    
    # Cosmology
    with open(os.path.join(info_folder, 'COSMOLOGY.json'), 'r') as file:
        cosmology_info = json.load(file)
    
    cosmology = pyccl.Cosmology(
        h=cosmology_info['H'],
        w0=cosmology_info['W0'],
        wa=cosmology_info['WA'],
        n_s=cosmology_info['NS'], 
        A_s=cosmology_info['AS'], 
        m_nu=cosmology_info['M_NU'],
        Neff=cosmology_info['N_EFF'],
        Omega_k=cosmology_info['OMEGA_K'], 
        Omega_b=cosmology_info['OMEGA_B'], 
        Omega_c=cosmology_info['OMEGA_CDM'],
        Omega_g=cosmology_info['OMEGA_GAMMA'], 
        mass_split = 'single', matter_power_spectrum = 'halofit', transfer_function = 'boltzmann_camb',
        extra_parameters = {'camb': {'kmax': 100, 'lmax': 5000, 'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.8}}
    )
    
    constant = 5e-14 / numpy.square(cosmology_info['H'])
    growth_factor = pyccl.background.growth_factor(cosmo=cosmology, a=1.0)
    rho_m = cosmology_info['OMEGA_M'] * pyccl.background.rho_x(cosmo=cosmology, a=1.0, species='critical', is_comoving=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    a_pivot = 0.5
    z_pivot = 0.5
    xi_pivot = 0.0
    a_grid = - constant * rho_m / growth_factor * a_pivot * numpy.power((1 + z_grid) / (1 + z_pivot), xi_pivot)
    
    alignment_info = {
        'A': a_grid.tolist(),
    }
    
    with open(os.path.join(info_folder, 'ALIGNMENT.json'), 'w') as file:
        json.dump(alignment_info, file, indent=4)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info Alignment')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(FOLDER)