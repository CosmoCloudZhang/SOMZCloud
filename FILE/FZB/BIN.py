import os
import time
import h5py
import numpy
import argparse
from rail import core


def main(folder, number):
    """
    Create a redshift bin.
    
    Arguments:
        path (str): The path to the base folder.
        length (int): The number of samples.
    
    Returns:
        numpy.ndarray: The redshift bin.
    """
    
    # Data
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Path
    data_path = os.path.join(path, 'DATA/')
    os.makedirs(os.path.join(data_path, 'BIN/'), exist_ok=True)
    
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)()
    
    # Bin
    bin_size = 5
    bin_lens = numpy.zeros((length, bin_size + 1))
    
    quantiles = numpy.linspace(0, 1, bin_size + 1)
    bin_source = numpy.zeros((length, bin_size + 1))
    
    for index in range(1, length + 1):
        start = time.time()
        
        estimate_name = os.path.join(data_path, 'FZB/FZB_ESTIMATE{}.hdf5'.format(index))
        estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)()
        
        # Lens
        z1_lens = 0.2
        z2_lens = 1.2
        bin_lens[index - 1, :] = numpy.linspace(z1_lens, z2_lens, bin_size + 1)
        
        z1_source = 0.0
        z2_source = 3.0
        z_mean = numpy.concatenate(estimator.mean())
        
        select_source = numpy.isfinite(z_mean) & (z1_source < z_mean) & (z_mean <= z2_source)
        
        bin_source[index - 1, :] = numpy.quantile(z_mean[select_source], quantiles)
        
        end = time.time()
        duration = (end - start) / 60
        print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    
    # Save
    with h5py.File(os.path.join(data_path, 'FZB/LENS/BIN.hdf5'), 'w') as file:
        file.create_dataset('bin', data=bin_lens)
    
    with h5py.File(os.path.join(data_path, 'FZB/SOURCE/BIN.hdf5'), 'w') as file:
        file.create_dataset('bin', data=bin_source)
    
    # Return
    return bin_lens, bin_source


if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='Lens Binning')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets')
    
    PATH = PARSE.parse_args().path
    LENGTH = PARSE.parse_args().length
    RESULT = main(PATH, LENGTH)