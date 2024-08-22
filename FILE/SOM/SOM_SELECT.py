
import os
import time
import h5py
import numpy
import argparse
from rail import core
import multiprocessing

def select(z_mean, z_lens, z_source, mag_source):
    """
    Select the samples based on the redshift and magnitude criteria.
    
    Parameters:
        z_mean (numpy.ndarray): The mean redshift values of the samples.
        z_lens (list): The redshift range of the lens samples.
        z_source (list): The redshift range of the source samples.
        mag_source (numpy.ndarray): The magnitude values of the source samples.
        
    Returns:
        tuple: The selected lens and source samples.
    """
    # Redshift
    z1_lens, z2_lens = z_lens
    z1_source, z2_source = z_source
    
    # Select
    slope = 4.0
    intercept = 18.0
    select_source = numpy.isfinite(z_mean) & (z1_source < z_mean) & (z_mean <= z2_source)
    select_lens = numpy.isfinite(z_mean) & (z1_lens < z_mean) & (z_mean <= z2_lens) & (mag_source < slope * z_mean + intercept)
    
    return select_lens, select_source


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
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Data
    data_path = os.path.join(path, 'DATA/')
    os.makedirs(os.path.join(data_path, 'SOM/LENS/'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'SOM/SOURCE/'), exist_ok=True)
    
    os.makedirs(os.path.join(data_path, 'SOM/LENS/LENS{}'.format(index)), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'SOM/SOURCE/SOURCE{}'.format(index)), exist_ok=True)
    
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    estimate_name = os.path.join(data_path, 'FZB/FZB_ESTIMATE{}.hdf5'.format(index))
    
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)()
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)()
    
    # Bin
    with h5py.File(os.path.join(data_path, 'BIN/LENS/LENS{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_lens = file['bin'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(data_path, 'BIN/SOURCE/SOURCE{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_source = file['bin'][:].astype(numpy.float32)
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    z_lens = [z1_lens, z2_lens]
    
    z1_source = 0.0
    z2_source = 3.0
    z_source = [z1_source, z2_source]
    
    z_mean = numpy.concatenate(estimator.mean())
    mag_source = test_data['photometry']['mag_i_lsst']
    
    # Save Select
    select_lens, select_source = select(z_mean, z_lens, z_source, mag_source)
    
    # Save Datasets
    for k in range(len(bin_lens) - 1): 
        select_bin_lens = select_lens & (bin_lens[k] <= z_mean) & (z_mean < bin_lens[k + 1])
        with h5py.File(os.path.join(data_path, 'SOM/LENS/LENS{}/SELECT{}.hdf5'.format(index, k + 1)), 'w') as file:
            for name in test_data.keys():
                file.create_group(name=name)
                for key, value in test_data[name].items():
                    file[name].create_dataset(key, data=value[select_bin_lens])
    
    for k in range(len(bin_source) - 1):
        select_bin_source = select_source & (bin_source[k] <= z_mean) & (z_mean < bin_source[k + 1])
        with h5py.File(os.path.join(data_path, 'SOM/SOURCE/SOURCE{}/SELECT{}.hdf5'.format(index, k + 1)), 'w') as file:
            for name in test_data.keys():
                file.create_group(name=name)
                for key, value in test_data[name].items():
                    file[name].create_dataset(key, data=value[select_bin_source])
    
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
