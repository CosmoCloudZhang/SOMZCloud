import os
import time
import h5py
import numpy
import argparse
from rail import core
import multiprocessing


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
    
    # Data store
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Data
    width = 1000
    bin_size = 5
    grid_size = 300
    
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Select
    for m in range(bin_size):
        select_name = os.path.join(data_path, 'SOM/LENS/LENS{}/SELECT{}.hdf5'.format(index, m + 1))
        cell_name = os.path.join(data_path, 'SOM/LENS/LENS{}/SOM_CELLID{}.hdf5'.format(index, m + 1))
        cluster_name = os.path.join(data_path, 'SOM/LENS/LENS{}/SOM_CELL_FILE{}.hdf5'.format(index, m + 1))
        
        cell_data = data_store.read_file(key='test_data', path=cell_name, handle_class=core.data.TableHandle)()
        select_data = data_store.read_file(key='test_data', path=select_name, handle_class=core.data.TableHandle)()
        cluster_data = data_store.read_file(key='test_data', path=cluster_name, handle_class=core.data.TableHandle)()
        
        cell_id = cell_data['cluster_ind'][:].astype(numpy.int32)
        cluster_id = cluster_data['uncovered_clusters'][:].astype(numpy.int32)
        redshift = select_data['photometry']['redshift'][:].astype(numpy.float32)
        
        z_pdf = numpy.zeros((width, grid_size))
        z_select = redshift[~numpy.isin(cell_id, cluster_id)]
        z_sample = numpy.random.choice(z_select, size=(width, z_select.size), replace=True)
        
        for k in range(width):
            z_pdf[k, :] = numpy.histogram(z_sample[k, :], bins=z_grid, density=True, range=(z1, z2))[0]
        
        with h5py.File(os.path.join(data_path, 'SOM/LENS/LENS{}/SOM_SUMMARIZE_SELECT{}.hdf5'.format(index, m + 1)), 'w') as file:
            
            file.create_group(name='meta')
            file.create_group(name='data')
            
            file['meta'].create_dataset('pdf_version', data=[0.0])
            file['meta'].create_dataset('pdf_name', data=['interp'])
            file['data'].create_dataset('yvals', data=z_pdf, dtype=numpy.float32)
            file['meta'].create_dataset('xvals', data=[z_grid], dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    return duration

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='SELECT')
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