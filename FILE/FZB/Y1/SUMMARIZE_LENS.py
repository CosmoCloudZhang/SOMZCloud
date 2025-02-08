import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, index, folder):
    '''
    Histogram of the spec redshifts of the lens samples
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_delta = (z2 - z1) / grid_size
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    z_bin = numpy.linspace(z1 - z_delta / 2, z2 + z_delta / 2, z_grid.size + 1)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_label = file['meta']['label'][:].astype(numpy.int32)
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_count = file['meta']['count'][:].astype(numpy.int32)
    som_size = len(combination_count)
    
    # Select
    with h5py.File(os.path.join(fzb_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        lens_bin = file['bin'][:].astype(numpy.float32)
        lens_select = file['select'][:].astype(bool)
    lens_bin_size = len(lens_bin) - 1
    sample_size = 1000
    
    # Summarize
    summarize_single = numpy.zeros((lens_bin_size, grid_size + 1))
    summarize_sample = numpy.zeros((lens_bin_size, sample_size, grid_size + 1))
    
    for m in range(lens_bin_size):
        select = lens_select[m, :]
        select_size = numpy.sum(select)
        
        with h5py.File(os.path.join(fzb_folder, '{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, index)), 'r') as file:
            z_pdf = ['data']['yvals'][select].astype(numpy.float32)
        
        application_label_select = application_label[select]
        application_count_select = numpy.bincount(application_label_select, minlength=som_size)
        application_weight_select = numpy.divide(application_count_select, combination_count, out=numpy.zeros(som_size), where=combination_count != 0)[application_label_select]
        
        summarize_single[m, :] = numpy.average(z_pdf, weights=application_weight_select, axis=0, returned=False, keepdims=False)
        
        for n in range(sample_size):
            z_sample = numpy.random.randint(0, sample_size, size=sample_size)
            summarize_data[n, :] = numpy.mean(z_pdf[z_sample, :], axis=0)
        summarize_single = numpy.mean(z_pdf, axis=0)
    
    # Save
    with h5py.File(os.path.join(fzb_folder, '{}/LENS/LENS{}/SUMMARIZE.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('single', data=lens_single, dtype=numpy.float32)
        file.create_dataset('sample', data=lens_sample, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Summarize')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the dataset')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(INDEX, FOLDER)
