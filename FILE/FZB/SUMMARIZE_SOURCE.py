import os
import time
import h5py
import numpy
import argparse
from rail import core
import multiprocessing


def summarize(index, folder):
    '''
    Summarize the redshift distribution of the lens samples.
    
    Arguments:
        index (int): The index of the dataset.
        folder (str): The base folder of the dataset.
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    
    # Load
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    with h5py.File(os.path.join(fzb_folder, 'SOURCE/BIN.hdf5'), 'r') as file:
        bin_source = file['bin'][index, :].astype(numpy.float32)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    z_bin = numpy.linspace(z1, z2, z_grid.size + 1)
    
    # Summarize
    width = 1000
    for m in range(len(bin_source) - 1):
        # Select
        select_name = os.path.join(fzb_folder, 'SOURCE/SOURCE{}/SELECT{}.hdf5'.format(index + 1, m + 1))
        select_data = data_store.read_file(key='select', path=select_name, handle_class=core.data.QPHandle)()
        
        z_pdf = select_data.pdf(z_grid)
        sample_size, pdf_size = z_pdf.shape
        summarize_data = numpy.zeros((width, pdf_size), dtype=numpy.float32)
        
        for n in range(width):
            z_index = numpy.random.randint(0, sample_size, size=sample_size)
            summarize_data[n, :] = numpy.mean(z_pdf[z_index, :], axis=0)
        summarize_single = numpy.mean(z_pdf, axis=0)
        
        # Save the single
        with h5py.File(os.path.join(fzb_folder, 'SOURCE/SOURCE{}/SINGLE{}.hdf5'.format(index + 1, m + 1)), 'w') as file:
            file.create_group('meta')
            file.create_group('data')
            
            file['meta'].create_dataset(name='pdf_name', data=['hist'])
            file['meta'].create_dataset(name='pdf_version', data=[0.0])
            file['meta'].create_dataset(name='bins', data=z_bin, dtype=numpy.float32)
            file['data'].create_dataset(name='pdfs', data=summarize_single, dtype=numpy.float32)
        
        # Save the data
        with h5py.File(os.path.join(fzb_folder, 'SOURCE/SOURCE{}/DATA{}.hdf5'.format(index + 1, m + 1)), 'w') as file:
            file.create_group('meta')
            file.create_group('data')
            
            file['meta'].create_dataset(name='pdf_name', data=['hist'])
            file['meta'].create_dataset(name='pdf_version', data=[0.0])
            file['meta'].create_dataset(name='bins', data=z_bin, dtype=numpy.float32)
            file['data'].create_dataset(name='pdfs', data=summarize_data, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    return duration


def main(count, number, folder):
    '''
    Summarize the redshift distribution of the source samples.
    
    Arguments:
        count (int): The count of the process.
        number (int): The number of the dataset.
        folder (str): The base folder of the dataset.
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Start
    start = time.time()
    
    # Multiprocessing
    size = number // count
    for chunk in range(size):
        print('CHUNK: {}'.format(chunk + 1))
        with multiprocessing.Pool(processes=count) as pool:
            pool.starmap(summarize, [(index, folder) for index in range(chunk * count, (chunk + 1) * count)])
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Total Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Summarize')
    PARSE.add_argument('--count', type=int, required=True, help='The count of the process')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the dataset')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    COUNT = PARSE.parse_args().count
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(COUNT, NUMBER, FOLDER)