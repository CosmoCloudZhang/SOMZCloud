import os
import time
import h5py
import numpy
import argparse
import multiprocessing


def bin(z1, z2, bin_size):
    """
    Create a redshift bin.
    
    Arguments:
        z1 (float): The minimum redshift.
        z2 (float): The maximum redshift.
        bin_size (int): The number
    
    Returns:
        numpy.ndarray: The redshift bin.
    """
    z_bin = numpy.linspace(z1, z2, bin_size + 1)
    return z_bin


def main(path, index):
    """
    Create a redshift bin.
    
    Arguments:
        path (str): The path to the base folder.
        index (int): The index of the redshift bin.
    
    Returns:
        numpy.ndarray: The redshift bin.
    """
    
    # Data
    start = time.time()
    data_path = os.path.join(path, 'DATA/')
    
    # Redshift
    bin_size = 5
    z1_lens = 0.2
    z2_lens = 1.2
    bin_lens = bin(z1_lens, z2_lens, bin_size)
    
    # Save
    os.makedirs(os.path.join(data_path, 'BIN/LENS/LENS{}'.format(index)), exist_ok=True)
    with h5py.File(os.path.join(data_path, 'BIN/LENS/LENS{}/BIN.hdf5'.format(index)), 'w') as file:
        file.create_dataset('bin', data=bin_lens)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    return bin_lens
    
if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='Lens Binning')
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