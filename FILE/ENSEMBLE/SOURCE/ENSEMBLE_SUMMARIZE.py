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
    summarize_data = numpy.zeros((width, pdf_size), dtype=numpy.float64)
    
    for n in range(width):
        z_index = numpy.random.randint(0, sample_size, size=sample_size)
        summarize_data[n, :] = numpy.mean(z_pdf[z_index, :], axis=0)
    summarize_single = numpy.mean(z_pdf, axis=0)
    return summarize_single, summarize_data


def main(path, index):
    """
    The main function to select the samples based on the redshift and magnitude criteria.
    
    Arguments:
        path (str): The path to the base folder.
        index (int): The index of the datasets.
    
    Returns:
        float: The duration of the selection.
    """
    # Data store
    start = time.time()
    data_path = os.path.join(path, 'DATA/')
    os.makedirs(os.path.join(data_path, 'ENSEMBLE/SOURCE'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'ENSEMBLE/SOURCE/SOURCE{}'.format(index)), exist_ok=True)
    
    # Data store
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Data
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    width = 1000
    bin_size = 5
    z_bin = (z_grid[+1:] + z_grid[:-1]) / 2.0
    
    # Summarize
    for m in range(bin_size):
        som_cell_name = os.path.join(data_path, 'SOM/SOURCE/SOURCE{}/SOM_CELLID{}.hdf5'.format(index, m + 1))
        fzb_select_name = os.path.join(data_path, 'FZB/SOURCE/SOURCE{}/FZB_SELECT{}.hdf5'.format(index, m + 1))
        som_cluster_name = os.path.join(data_path, 'SOM/SOURCE/SOURCE{}/SOM_CELL_FILE{}.hdf5'.format(index, m + 1))
        
        som_cell_data = data_store.read_file(key='test_data', path=som_cell_name, handle_class=core.data.TableHandle)()
        fzb_select_data = data_store.read_file(key='select_data', path=fzb_select_name, handle_class=core.data.QPHandle)()
        som_cluster_data = data_store.read_file(key='test_data', path=som_cluster_name, handle_class=core.data.TableHandle)()
        
        som_cell_id = som_cell_data['cluster_ind'][:].astype(numpy.int32)
        som_cluster_id = som_cluster_data['uncovered_clusters'][:].astype(numpy.int32)
        
        z_size = numpy.minimum(len(som_cell_id), len(fzb_select_data.mean()))
        som_select = ~numpy.isin(som_cell_id, som_cluster_id)[:z_size]
        z_pdf = fzb_select_data.pdf(z_grid)[:z_size, :][som_select, :]
        summarize_single, summarize_data = summarize(z_pdf, width)
        
        # Save the data
        with h5py.File(os.path.join(data_path, 'ENSEMBLE/SOURCE/SOURCE{}/SINGLE{}.hdf5'.format(index, m + 1)), 'w') as file:
            file.create_group('meta')
            file.create_group('data')
            
            file['meta'].create_dataset(name='pdf_name', data=['hist'])
            file['meta'].create_dataset(name='pdf_version', data=[0.0])
            file['meta'].create_dataset(name='bins', data=z_bin, dtype=numpy.float64)
            file['data'].create_dataset(name='pdfs', data=summarize_single, dtype=numpy.float64)
        del summarize_single
        
        with h5py.File(os.path.join(data_path, 'ENSEMBLE/SOURCE/SOURCE{}/SUMMARIZE{}.hdf5'.format(index, m + 1)), 'w') as file:
            file.create_group('meta')
            file.create_group('data')
            
            file['meta'].create_dataset(name='pdf_name', data=['hist'])
            file['meta'].create_dataset(name='pdf_version', data=[0.0])
            file['meta'].create_dataset(name='bins', data=z_bin, dtype=numpy.float64)
            file['data'].create_dataset(name='pdfs', data=summarize_data, dtype=numpy.float64)
        del summarize_data
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    return duration

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='Ensemble Summarize')
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
            RESULT = POOL.starmap(main, [(PATH, INDEX) for INDEX in range(CHUNK * NUMBER + 1, (CHUNK + 1) * NUMBER + 1)])