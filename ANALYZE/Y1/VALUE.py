import os
import h5py
import time
import numpy
import scipy
import argparse


def main(tag, name, label, folder):
    '''
    This function is used to analyze the information of the dataset
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    print('Name: {}, Label: {}'.format(name, label))
    
    # Path
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    os.makedirs(os.path.join(analyze_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(analyze_folder, '{}/VALUE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(analyze_folder, '{}/VALUE/{}/'.format(tag, name)), exist_ok=True)
    
    # Summarize
    with h5py.File(os.path.join(synthesize_folder, '{}/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta'].keys()}
        
        data_lens = file['lens']['data'][...]
        data_source = file['source']['data'][...]
        
        average_lens = file['lens']['average'][...]
        average_source = file['source']['average'][...]
    
    # Redshift
    z_grid = meta['z_grid']
    
    # Mu
    mu_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * data_lens, axis=2)
    mu_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * data_source, axis=2)
    
    # Eta
    eta_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - mu_lens[:, :, numpy.newaxis]) * data_lens, axis=2))
    eta_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - mu_source[:, :, numpy.newaxis]) * data_source, axis=2))
    
    # Average
    average_mu_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * average_lens, axis=1)
    average_mu_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * average_source, axis=1)
    
    average_eta_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - average_mu_lens[:, numpy.newaxis]) * average_lens, axis=1))
    average_eta_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - average_mu_source[:, numpy.newaxis]) * average_source, axis=1))
    
    # Sigma
    sigma_mu_lens = numpy.std(mu_lens, axis=0) / (1 + average_mu_lens)
    sigma_mu_source = numpy.std(mu_source, axis=0) / (1 + average_mu_source)
    
    sigma_eta_lens = numpy.std(eta_lens, axis=0) / (1 + average_mu_lens)
    sigma_eta_source = numpy.std(eta_source, axis=0) / (1 + average_mu_source)
    
    # Rho
    rho_mu_lens = numpy.corrcoef(mu_lens, rowvar=False)
    rho_mu_source = numpy.corrcoef(mu_source, rowvar=False)
    
    rho_eta_lens = numpy.corrcoef(eta_lens, rowvar=False)
    rho_eta_source = numpy.corrcoef(eta_source, rowvar=False)
    
    # Save
    with h5py.File(os.path.join(analyze_folder, '{}/VALUE/{}/{}.hdf5'.format(tag, name, label)), 'w') as file:
        file.create_group('meta')
        for key in meta.keys():
            file['meta'].create_dataset(key, data=meta[key], dtype=meta[key].dtype)
        
        file.create_group('lens')
        file['lens'].create_dataset('mu', data=mu_lens)
        file['lens'].create_dataset('eta', data=eta_lens)
        
        file['lens'].create_dataset('average_mu', data=average_mu_lens)
        file['lens'].create_dataset('average_eta', data=average_eta_lens)
        
        file['lens'].create_dataset('sigma_mu', data=sigma_mu_lens)
        file['lens'].create_dataset('sigma_eta', data=sigma_eta_lens)
        
        file['lens'].create_dataset('rho_mu', data=rho_mu_lens)
        file['lens'].create_dataset('rho_eta', data=rho_eta_lens)
        
        file.create_group('source')
        file['source'].create_dataset('mu', data=mu_source)
        file['source'].create_dataset('eta', data=eta_source)
        
        file['source'].create_dataset('average_mu', data=average_mu_source)
        file['source'].create_dataset('average_eta', data=average_eta_source)
        
        file['source'].create_dataset('sigma_mu', data=sigma_mu_source)
        file['source'].create_dataset('sigma_eta', data=sigma_eta_source)
        
        file['source'].create_dataset('rho_mu', data=rho_mu_source)
        file['source'].create_dataset('rho_eta', data=rho_eta_source)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Analyze Value')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, LABEL, FOLDER)