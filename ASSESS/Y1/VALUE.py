import os
import h5py
import time
import numpy
import scipy
import argparse


def main(tag, name, label, index, folder):
    '''
    This function is used to analyze the information of the dataset
        
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
        label (str): The label of the configuration
        index (int): The index of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    print('Name: {} Label: {} Index: {}'.format(name, label, index))
    
    # Path
    assess_folder = os.path.join(folder, 'ASSESS/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    os.makedirs(os.path.join(assess_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(assess_folder, '{}/VALUE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(assess_folder, '{}/VALUE/{}/'.format(tag, name)), exist_ok=True)
    os.makedirs(os.path.join(assess_folder, '{}/VALUE/{}/{}/'.format(tag, name, label)), exist_ok=True)
    
    # Summarize Lens
    with h5py.File(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}/{}.hdf5'.format(tag, name, index, label)), 'r') as file:
        data_lens = file['ensemble']['data'][...]
        average_lens = file['ensemble']['average'][...]
        meta_lens = {key: file['meta'][key][...] for key in file['meta'].keys()}
    
    z1 = meta_lens['z1']
    z2 = meta_lens['z2']
    z_grid = meta_lens['z_grid']
    grid_size = meta_lens['grid_size']
    
    bin_lens = meta_lens['bin']
    bin_lens_size = meta_lens['bin_size']
    data_lens = numpy.transpose(data_lens, (1, 0, 2))
    
    # Summarize Source
    with h5py.File(os.path.join(summarize_folder, '{}/{}/SOURCE/SOURCE{}/{}.hdf5'.format(tag, name, index, label)), 'r') as file:
        data_source = file['ensemble']['data'][...]
        average_source = file['ensemble']['average'][...]
        meta_source = {key: file['meta'][key][...] for key in file['meta'].keys()}
    
    z1 = meta_source['z1']
    z2 = meta_source['z2']
    z_grid = meta_source['z_grid']
    grid_size = meta_source['grid_size']
    
    bin_source = meta_source['bin']
    bin_source_size = meta_source['bin_size']
    data_source = numpy.transpose(data_source, (1, 0, 2))
    
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
    
    # Save
    with h5py.File(os.path.join(assess_folder, '{}/VALUE/{}/{}/DATA{}.hdf5'.format(tag, name, label, index)), 'w') as file:
        file.create_group('meta')
        file['meta'].create_dataset('z1', data=z1, dtype=numpy.float32)
        file['meta'].create_dataset('z2', data=z2, dtype=numpy.float32)
        file['meta'].create_dataset('z_grid', data=z_grid, dtype=numpy.float32)
        file['meta'].create_dataset('grid_size', data=grid_size, dtype=numpy.int32)
        
        file['meta'].create_dataset('bin_lens', data=bin_lens, dtype=numpy.float32)
        file['meta'].create_dataset('bin_lens_size', data=bin_lens_size, dtype=numpy.int32)
        
        file['meta'].create_dataset('bin_source', data=bin_source, dtype=numpy.float32)
        file['meta'].create_dataset('bin_source_size', data=bin_source_size, dtype=numpy.int32)
        
        file.create_group('lens')
        file['lens'].create_dataset('mu', data=mu_lens)
        file['lens'].create_dataset('eta', data=eta_lens)
        
        file['lens'].create_dataset('average_mu', data=average_mu_lens)
        file['lens'].create_dataset('average_eta', data=average_eta_lens)
        
        file['lens'].create_dataset('sigma_mu', data=sigma_mu_lens)
        file['lens'].create_dataset('sigma_eta', data=sigma_eta_lens)
        
        file.create_group('source')
        file['source'].create_dataset('mu', data=mu_source)
        file['source'].create_dataset('eta', data=eta_source)
        
        file['source'].create_dataset('average_mu', data=average_mu_source)
        file['source'].create_dataset('average_eta', data=average_eta_source)
        
        file['source'].create_dataset('sigma_mu', data=sigma_mu_source)
        file['source'].create_dataset('sigma_eta', data=sigma_eta_source)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Assess Value')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    LABEL = PARSE.parse_args().label
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, LABEL, INDEX, FOLDER)