import os
import time
import h5py
import numpy
import scipy
import argparse

def main(tag, index, folder):
    '''
    Histogram of the spectroscopic redshifts of the lens samples
    
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
    
    os.makedirs(os.path.join(fzb_folder, '{}/LENS/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(fzb_folder, '{}/LENS/LENS{}'.format(tag, index)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_size = len(file['photometry']['redshift'][...])
    
    # Select
    with h5py.File(os.path.join(fzb_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_lens = file['bin_lens'][...]
    
    with h5py.File(os.path.join(fzb_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        select_lens = file['select'][...]
    
    # Size
    sample_size = 100
    bin_lens_size = len(bin_lens) - 1
    
    # Lens
    single_lens = numpy.zeros((bin_lens_size, grid_size + 1))
    sample_lens = numpy.zeros((bin_lens_size, sample_size, grid_size + 1))
    
    # Chunk
    chunk_size = 10000
    estimator = h5py.File(os.path.join(fzb_folder, '{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, index)), 'r')
    
    # Loop
    for m in range(bin_lens_size):
        # Select
        select = select_lens[m, :] 
        select_size = numpy.sum(select)
        select_indices = numpy.arange(application_size)[select]
        
        histogram_single = numpy.zeros((grid_size + 1))
        histogram_sample = numpy.zeros((sample_size, grid_size + 1))
        
        sample_weight = numpy.ones((sample_size, select_size))
        for k in range(sample_size):
            sample_indices = numpy.random.choice(numpy.arange(select_size), size=select_size, replace=True)
            sample_weight[k, :] = numpy.bincount(sample_indices, minlength=select_size)
        
        # Loop
        for n in range(select_size // chunk_size + 1):
            # PDF
            begin = n * chunk_size
            end = min((n + 1) * chunk_size, application_size)
            z_pdf = estimator['data']['yvals'][select_indices[begin: end]].astype(numpy.float32)
            
            # Histogram
            histogram_single = histogram_single + numpy.sum(z_pdf, axis=0)
            histogram_sample = histogram_sample + numpy.sum(z_pdf[numpy.newaxis, :, :] * sample_weight[:, begin: end][:, :, numpy.newaxis], axis=1)
        
        # Normalize
        single_lens[m, :] = histogram_single / scipy.integrate.trapezoid(x=z_grid, y=histogram_single, axis=0)
        sample_lens[m, :, :] = histogram_sample / scipy.integrate.trapezoid(x=z_grid, y=histogram_sample, axis=1)[:, numpy.newaxis]
    
    # Save
    with h5py.File(os.path.join(fzb_folder, '{}/LENS/LENS{}/SUMMARIZE.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('single', data=single_lens, dtype=numpy.float32)
        file.create_dataset('sample', data=sample_lens, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Summarize')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)
