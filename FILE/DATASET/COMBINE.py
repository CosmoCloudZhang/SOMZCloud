import os
import h5py
import time
import numpy
import argparse

def main(number, folder):
    '''
    Combine the datasets
    
    Arguments:
        number (int): The number of datasets
        folder (str): The base folder of the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    for index in range(1, number + 1):
        print('Index: {}'.format(index))
        
        # Selection
        with h5py.File(os.path.join(dataset_folder, 'SELECTION/DATA{}.hdf5'.format(index)), 'r') as file:
            selection_data = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
            
        # Augmentation
        with h5py.File(os.path.join(dataset_folder, 'AUGMENTATION/DATA{}.hdf5'.format(index)), 'r') as file:
            augmentation_data = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
        
        # Combine
        data = {'photometry': {}}
        data['photometry']['redshift'] = numpy.append(selection_data['redshift'], augmentation_data['redshift'])
        
        band_list = ['u_lsst', 'g_lsst', 'r_lsst', 'i_lsst', 'z_lsst', 'y_lsst']
        for band in band_list:
            
            data['photometry']['mag_{}'.format(band)] = numpy.append(selection_data['mag_{}'.format(band)], augmentation_data['mag_{}'.format(band)])
            data['photometry']['mag_err_{}'.format(band)] = numpy.append(selection_data['mag_err_{}'.format(band)], augmentation_data['mag_err_{}'.format(band)])
        
        # Save
        os.makedirs(dataset_folder, exist_ok=True)
        os.makedirs(os.path.join(dataset_folder, 'COMBINATION'), exist_ok=True)
        
        with h5py.File(os.path.join(dataset_folder, 'COMBINATION/DATA{}.hdf5'.format(index)), 'w') as file:
            file.create_group('photometry')
            for key, value in data['photometry'].items():
                file['photometry'].create_dataset(key, data=value)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Combination datasets')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(NUMBER, FOLDER)