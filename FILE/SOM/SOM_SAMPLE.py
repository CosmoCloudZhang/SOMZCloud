import os
import time
import h5py
import numpy
import argparse
from rail import core
import multiprocessing


def main(path, index):
    # Data Store
    start = time.time()
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Path
    data_path = os.path.join(path, 'DATA/')
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    train_name = os.path.join(data_path, 'SAMPLE/TRAIN_SAMPLE{}.hdf5'.format(index))
    
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)
    train_data = data_store.read_file(key='train_data', path=train_name, handle_class=core.data.TableHandle)
    
    som_name = os.path.join(data_path, 'SOM/SOM_SAMPLE{}.hdf5'.format(index))
    with h5py.File(som_name, 'w') as file:
        file.create_group('photometry')
        for key in test_data()['photometry'].keys():
            test_value = test_data()['photometry'][key]
            train_value = train_data()['photometry'][key]
            
            value = numpy.concatenate([test_value, train_value]).astype(numpy.float32)
            file['photometry'].create_dataset(key, data=value)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
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
    
    # Multiprocessing
    SIZE = LENGTH // NUMBER
    for CHUNK in range(SIZE):
        print('CHUNK: {}'.format(CHUNK + 1))
        with multiprocessing.Pool(processes=NUMBER) as POOL:
            POOL.starmap(main, [(PATH, INDEX) for INDEX in range(CHUNK * NUMBER + 1, (CHUNK + 1) * NUMBER + 1)])