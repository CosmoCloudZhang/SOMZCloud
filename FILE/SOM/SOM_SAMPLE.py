import os
import time
import h5py
import numpy
import argparse
from rail import core


def main(path, length):
    """
    The main function to create the som sample.
    
    Arguments:
        path (str): The path to the
        length (int): The length of the train datasets
    
    Returns:
        None
    """
    # Data Store
    start = time.time()
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Path
    data_path = os.path.join(path, 'DATA/')
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)()
    
    spec_data = {'photometry': {}}
    for index in range(1, length + 1):
        
        train_name = os.path.join(data_path, 'SAMPLE/TRAIN_SAMPLE{}.hdf5'.format(index))
        train_data = data_store.read_file(key='train_data', path=train_name, handle_class=core.data.TableHandle)()
        
        for train_key, train_value in train_data['photometry'].items():
            if train_key in spec_data['photometry'].keys():
                spec_data['photometry'][train_key] = numpy.concatenate([spec_data['photometry'][train_key], train_value])
            else:
                spec_data['photometry'][train_key] = train_value
    spec_size = spec_data['photometry']['redshift'].size
    spec_index = numpy.random.choice(numpy.arange(spec_size), size=spec_size // length, replace=False)
    
    # Save
    phot_name = os.path.join(data_path, 'SOM/SOM_SAMPLE.hdf5')
    with h5py.File(phot_name, 'w') as file:
        file.create_group('photometry')
        
        for test_key, test_value in test_data['photometry'].items():
            spec_value = spec_data['photometry'][test_key][spec_index]
            value = numpy.concatenate([test_value, spec_value])
            
            if test_key in file['photometry'].keys():
                file['photometry'][test_key][...] = value
            else:
                file['photometry'].create_dataset(test_key, data=value, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Augmentation sample.')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets')
    
    PATH = PARSE.parse_args().path
    LENGTH = PARSE.parse_args().length
    
    RESULT = main(PATH, LENGTH)