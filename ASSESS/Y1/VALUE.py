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
    print('Name: {}, Label: {}'.format(name, label))
    
    # Path
    assess_folder = os.path.join(folder, 'ASSESS/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    os.makedirs(os.path.join(assess_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(assess_folder, '{}/{}/'.format(tag, name)), exist_ok=True)
    os.makedirs(os.path.join(assess_folder, '{}/{}/VALUE/'.format(tag, name)), exist_ok=True)
    os.makedirs(os.path.join(assess_folder, '{}/{}/VALUE/{}/'.format(tag, name, label)), exist_ok=True)
    
    # Summarize Lens
    with h5py.File(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}/{}.hdf5'.format(tag, name, index, label)), 'r') as file:
        data_lens = file['data'][...]
        average_lens = file['average'][...]
    bin_lens_size, z_size = average_lens.shape
    data_lens = numpy.transpose(data_lens, (1, 0, 2))
    
    # Summarize Source
    with h5py.File(os.path.join(summarize_folder, '{}/{}/SOURCE/SOURCE{}/{}.hdf5'.format(tag, name, index, label)), 'r') as file:
        data_source = file['data'][...]
        average_source = file['average'][...]
    bin_source_size, z_size = average_source.shape
    data_source = numpy.transpose(data_source, (1, 0, 2))
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, z_size)
    
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
    with h5py.File(os.path.join(assess_folder, '{}/{}/VALUE/{}/DATA{}.hdf5'.format(tag, name, label, index)), 'w') as file:
        file.create_group('meta')
        file['meta'].create_dataset('z1', data=z1, dtype=numpy.float32)
        file['meta'].create_dataset('z2', data=z2, dtype=numpy.float32)
        file['meta'].create_dataset('z_size', data=z_size, dtype=numpy.int32)
        file['meta'].create_dataset('z_grid', data=z_grid, dtype=numpy.float32)
        file['meta'].create_dataset('bin_lens_size', data=bin_lens_size, dtype=numpy.int32)
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