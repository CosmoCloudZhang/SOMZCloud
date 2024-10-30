import os
import h5py
import time
import numpy
import argparse
import GCRCatalogs
import multiprocessing

def sample(path, index):
    data = []
    return data


def main(path, length, number):
    
    start = time.time()
    print(GCRCatalogs.get_available_catalog_names())
    
    '''
    size = length // number
    for chunk in range(size):
        print('Chunk: {}'.format(chunk + 1))
        with multiprocessing.Pool(processes=number) as pool:
            pool.starmap(sample, [(path, index) for index in range(chunk * number + 1, (chunk + 1) * number + 1)])
    '''
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Augmentation sample.')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the processes')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets')
    
    PATH = PARSE.parse_args().path
    NUMBER = PARSE.parse_args().number
    LENGTH = PARSE.parse_args().length
    
    # Output
    OUTPUT = main(PATH, LENGTH, NUMBER)