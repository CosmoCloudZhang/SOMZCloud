import os
import time
import h5py
import numpy
import argparse

def main(path, width, length, number):
    start = time.time()
    
    # Data
    height = length * width
    data_path = os.path.join(path, 'DATA/')
    
    lens_count = numpy.zeros((height, number), dtype=numpy.float32)
    lens_data = numpy.zeros((height, number, width), dtype=numpy.float32)
    
    source_count = numpy.zeros((height, number), dtype=numpy.float32)
    source_data = numpy.zeros((height, number, width), dtype=numpy.float32)
    
    for m in range(length):
        
        lens_name = os.path.join(data_path, 'LENS/LENS{}/SELECT.hdf5'.format(m + 1))
        source_name = os.path.join(data_path, 'SOURCE/SOURCE{}/SELECT.hdf5'.format(m + 1))
        
        with h5py.File(lens_name.format(m), 'r') as file:
            lens_count[m * width: (m + 1) * width, :] = file['count'][:].astype(numpy.float32)
            lens_data[m * width: (m + 1) * width, :] = file['data'][:].astype(numpy.float32)
        
        with h5py.File(source_name.format(m), 'r') as file:
            source_count[m * width: (m + 1) * width, :] = file['count'][:].astype(numpy.float32)
            source_data[m * width: (m + 1) * width, :] = file['data'][:].astype(numpy.float32)
    
    # Save
    lens = {'data': lens_data, 'count': lens_count}
    source = {'data': source_data, 'count': source_count}
    
    with h5py.File(os.path.join(data_path, 'WEIGHT/LENS.hdf5'), 'w') as file:
        for key, value in lens.items():
            file.create_dataset(key, data=value)
    
    with h5py.File(os.path.join(data_path, 'WEIGHT/SOURCE.hdf5'), 'w') as file:
        for key, value in source.items():
            file.create_dataset(key, data=value)
    
    # Return
    end = time.time()
    print('Time: {:.2f} minutes'.format((end - start) / 60))
    return lens, source

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Informer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--width', type=int, required=True, help='The width of the train datasets')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the tomography bins')
    
    PATH = PARSE.parse_args().path
    WIDTH = PARSE.parse_args().width
    LENGTH = PARSE.parse_args().length
    NUMBER = PARSE.parse_args().number
    LENS, SOURCE = main(PATH, WIDTH, LENGTH, NUMBER)