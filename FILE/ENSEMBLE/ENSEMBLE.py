import os
import time
import h5py
import numpy
import argparse

def main(path, width, length, number):
    start = time.time()
    
    # Data
    z_size = 300
    weight_size = 1
    height = length * width
    data_path = os.path.join(path, 'DATA/')
    
    lens_count = numpy.zeros((height, number), dtype=numpy.float32)
    lens_sample = numpy.zeros((height, number, z_size), dtype=numpy.float32)
    
    source_count = numpy.zeros((height, number), dtype=numpy.float32)
    source_sample = numpy.zeros((height, number, z_size), dtype=numpy.float32)
    
    for m in range(length):
        
        lens_name = os.path.join(data_path, 'LENS/LENS{}/SELECT.hdf5'.format(m + 1))
        source_name = os.path.join(data_path, 'SOURCE/SOURCE{}/SELECT.hdf5'.format(m + 1))
        
        with h5py.File(lens_name.format(m), 'r') as file:
            lens_count[m * width: (m + 1) * width, :] = file['count'][:].astype(numpy.float32)
            lens_sample[m * width: (m + 1) * width, :] = file['sample'][:].astype(numpy.float32)
        
        with h5py.File(source_name.format(m), 'r') as file:
            source_count[m * width: (m + 1) * width, :] = file['count'][:].astype(numpy.float32)
            source_sample[m * width: (m + 1) * width, :] = file['sample'][:].astype(numpy.float32)
    
    lens_weight_count = numpy.zeros((height, number), dtype=numpy.float32)
    lens_weight_data = numpy.zeros((height, number, z_size), dtype=numpy.float32)
    
    source_weight_count = numpy.zeros((height, number), dtype=numpy.float32)
    source_weight_data = numpy.zeros((height, number, z_size), dtype=numpy.float32)
    
    for m in range(height):
        length_index = numpy.random.choice(numpy.arange(length), weight_size, replace=False)
        width_index = numpy.random.choice(numpy.arange(width), weight_size, replace=False)
        weight_index = length_index * width + width_index
        
        lens_weight_count[m, :] = numpy.sum(lens_count[weight_index, :], axis=0)
        lens_weight_data[m, :, :] = numpy.sum(lens_sample[weight_index, :, :], axis=0)
        
        source_weight_count[m, :] = numpy.sum(source_count[weight_index, :], axis=0)
        source_weight_data[m, :, :] = numpy.sum(source_sample[weight_index, :, :], axis=0)
    
    # Save
    lens = {'data': lens_weight_data, 'count': lens_weight_count}
    source = {'data': source_weight_data, 'count': source_weight_count}
    
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