import os
import json
import time
import argparse


def main(folder):
    '''
    Store the fiducial values of cosmological parameters
    
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
    cosmology_info = {
        'WA': 0.0, 
        'W0': -1.0,
        'H': 0.6736,
        'N_NR': 1.0,
        'N_RUN': 0.0,
        'M_NU': 0.06,
        'NS': 0.9649,
        'N_UR': 2.0328,
        'OMEGA_K': 0.0, 
        'AS': 2.083e-09, 
        'N_EFF': 3.0328,
        'OMEGA_DE': 0.684727161632854, 
        'OMEGA_H': 0.3137721026738836,
        'OMEGA_B': 0.04930169232854376,
        'OMEGA_CDM': 0.2644704103453399, 
        'OMEGA_M': 0.315193355726963160,
        'OMEGA_R': 7.948264018270915e-05,  
        'OMEGA_GAMMA': 5.45023999774456e-05, 
        'OMEGA_NU_NR': 0.0014212530530795793,
        'OMEGA_NU_UR': 2.498024020526355e-05,
    }
    
    with open(os.path.join(info_folder, 'COSMOLOGY.json'), 'w') as file:
        json.dump(cosmology_info, file, indent=4)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info Cosmology')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(FOLDER)