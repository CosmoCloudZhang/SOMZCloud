import os
import time
import h5py
import numpy
import argparse
from rail import core


def main(index, folder):
    '''
    Summarize the redshift distribution of the source samples.
    
    Arguments:
        index (int): The index of the dataset.
        folder (str): The base folder of the dataset.
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    print('Index:{}'.format(index))
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    
    # Load
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    with h5py.File(os.path.join(fzb_folder, 'SOURCE/SOURCE{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_source = file['bin'][:].astype(numpy.float32)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Summarize
    width = 1000
    for m in range(1, len(bin_source)):
        # Sample
        sample_name = os.path.join(fzb_folder, 'SOURCE/SOURCE{}/SAMPLE{}.hdf5'.format(index, m))
        sample_data = data_store.read_file(key='sample', path=sample_name, handle_class=core.data.QPHandle)()
        
        z_pdf = sample_data.pdf(z_grid)
        sample_size, pdf_size = z_pdf.shape
        summarize_data = numpy.zeros((width, pdf_size), dtype=numpy.float32)
        
        for n in range(width):
            z_index = numpy.random.randint(0, sample_size, size=sample_size)
            summarize_data[n, :] = numpy.mean(z_pdf[z_index, :], axis=0)
        summarize_single = numpy.mean(z_pdf, axis=0)
        
        # Save the single
        with h5py.File(os.path.join(fzb_folder, 'SOURCE/SOURCE{}/SINGLE{}.hdf5'.format(index, m)), 'w') as file:
            file.create_group('meta')
            file.create_group('data')
            
            file['meta'].create_dataset(name='pdf_name', data=['hist'])
            file['meta'].create_dataset(name='pdf_version', data=[0.0])
            file['meta'].create_dataset(name='bins', data=z_grid, dtype=numpy.float32)
            file['data'].create_dataset(name='pdfs', data=summarize_single, dtype=numpy.float32)
        
        # Save the data
        with h5py.File(os.path.join(fzb_folder, 'SOURCE/SOURCE{}/SUMMARIZE{}.hdf5'.format(index, m)), 'w') as file:
            file.create_group('meta')
            file.create_group('data')
            
            file['meta'].create_dataset(name='pdf_name', data=['hist'])
            file['meta'].create_dataset(name='pdf_version', data=[0.0])
            file['meta'].create_dataset(name='bins', data=z_grid, dtype=numpy.float32)
            file['data'].create_dataset(name='pdfs', data=summarize_data, dtype=numpy.float32)
    
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
