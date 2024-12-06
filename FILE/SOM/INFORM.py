import os
import time
import h5py
import yaml
import numpy
import argparse


def main(number, folder):
    '''
    Main function for the sampling of the datasets.
    
    Arguments:
        number (int) : the number of the datasets
        folder (str) : the folder of the datasets
    
    Returns:
        duration (float) : the duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    som_folder = os.path.join(folder, 'SOM/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Data
    dataset = {}
    for index in range(1, number + 1):
        print('Index: {}'.format(index))
        with h5py.File(os.path.join(dataset_folder, 'APPLICATION/DATA{}.hdf5'.format(index)), 'r') as file:
            size = file['photometry']['redshift'][:].astype(numpy.float32).size
            indices = numpy.random.choice(numpy.arange(size), size=size // number, replace=False)
            for key in file['photometry'].keys():
                if key in dataset.keys():
                    dataset[key] = numpy.append(dataset[key], file['photometry'][key][:].astype(numpy.float32)[indices])
                else:
                    dataset[key] = file['photometry'][key][:].astype(numpy.float32)[indices]
    # Save
    os.makedirs(os.path.join(som_folder, 'INFORM/'), exist_ok=True)
    with h5py.File(os.path.join(som_folder, 'INFORM/INFORM.hdf5'), 'w') as file:
        file.create_group('photometry')
        for key, value in dataset.items():
            file['photometry'].create_dataset(key, data=value, dtype=numpy.float32)
    
    # Config
    config = {
        'INFORM': {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'model': 'input_model',
            }, 
            'seed': 0, 
            'std_coeff': 0.5, 
            'mag_limits': 30.0,
            'maptype': 'toroid', 
            'nondetect_val': 30.0, 
            'grid_type': 'hexagonal', 
            'ref_band': 'mag_i_lsst',
            'column_usage': 'colors',
            'som_learning_rate': 0.5, 
            'hdf5_groupname': 'photometry', 
            'n_rows': 100, 'n_columns': 100, 
            'bands': [
                'mag_u_lsst', 
                'mag_g_lsst', 
                'mag_r_lsst', 
                'mag_i_lsst', 
                'mag_z_lsst', 
                'mag_y_lsst'
            ], 
            'err_bands': [
                'mag_err_u_lsst', 
                'mag_err_g_lsst', 
                'mag_err_r_lsst', 
                'mag_err_i_lsst', 
                'mag_err_z_lsst', 
                'mag_err_y_lsst'
            ]
        }
    }
    
    # Save
    os.makedirs(os.path.join(som_folder, 'INFORM'), exist_ok=True)
    config_name = os.path.join(som_folder, 'INFORM/INFORM.yaml')
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='SOM Informer')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(NUMBER, FOLDER)