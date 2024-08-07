import os
import time
import h5py
import numpy
import argparse
from rail import core
import multiprocessing


def summarize(z_pdf, width):
    """
    The function to summarize the data.
    
    Arguments:
        select_data (QPHandle): The data handle.
        z_data (numpy.ndarray): The redshift data.
    
    Returns:
        numpy.ndarray: The summarized data.
    """
    sample_size, pdf_size = z_pdf.shape
    summarize_data = numpy.zeros((width, pdf_size), dtype=numpy.float32)
    
    for n in range(width):
        z_index = numpy.random.randint(0, sample_size, size=sample_size)
        summarize_data[n, :] = numpy.mean(z_pdf[z_index, :], axis=0)
    summarize_single = numpy.mean(z_pdf, axis=0)
    return summarize_single, summarize_data


def main(path, index):
    """
    The main function to resample the redshift and get ensemble distributions.
    
    Arguments:
        path (str): The path to the base folder.
        index (int): The index of the datasets.
    
    Returns:
        float: The duration of the selection.
    """
    # Data store
    start = time.time()
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Path
    data_path = os.path.join(path, 'DATA/')
    os.makedirs(os.path.join(data_path, 'FZB/LENS'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'FZB/LENS/LENS{}'.format(index)), exist_ok=True)
    
    # Bin
    with h5py.File(os.path.join(data_path, 'BIN/LENS/LENS{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_lens = file['bin'][:].astype(numpy.float32)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    z_bin = numpy.linspace(z1, z2, z_grid.size + 1)
    
    # Summarize
    width = 1000
    for m in range(len(bin_lens) - 1):
        select_name = os.path.join(data_path, 'FZB/LENS/LENS{}/FZB_SELECT{}.hdf5'.format(index, m + 1))
        select_data = data_store.read_file(key='select_data', path=select_name, handle_class=core.data.QPHandle)()
        
        z_pdf = select_data.pdf(z_grid)
        summarize_single, summarize_data = summarize(z_pdf, width)
        del z_pdf, select_name, select_data
        
        # Save the single
        with h5py.File(os.path.join(data_path, 'FZB/LENS/LENS{}/FZB_SINGLE{}.hdf5'.format(index, m + 1)), 'w') as file:
            file.create_group('meta')
            file.create_group('data')
            
            file['meta'].create_dataset(name='pdf_name', data=['hist'])
            file['meta'].create_dataset(name='pdf_version', data=[0.0])
            file['meta'].create_dataset(name='bins', data=z_bin, dtype=numpy.float32)
            file['data'].create_dataset(name='pdfs', data=summarize_single, dtype=numpy.float32)
        del summarize_single
        # Save the data
        with h5py.File(os.path.join(data_path, 'FZB/LENS/LENS{}/FZB_SUMMARIZE{}.hdf5'.format(index, m + 1)), 'w') as file:
            file.create_group('meta')
            file.create_group('data')
            
            file['meta'].create_dataset(name='pdf_name', data=['hist'])
            file['meta'].create_dataset(name='pdf_version', data=[0.0])
            file['meta'].create_dataset(name='bins', data=z_bin, dtype=numpy.float32)
            file['data'].create_dataset(name='pdfs', data=summarize_data, dtype=numpy.float32)
        del summarize_data
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    return duration


if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Summarize')
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