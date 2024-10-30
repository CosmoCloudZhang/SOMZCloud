import os
import h5py
import time
import numpy
import argparse
import multiprocessing

def sample(path, index):
    data = []
    return data


def main(path, folder, length, number):
    
    start = time.time()
    data_path = os.path.join(path, 'DATA/')
    
    magnitude =  {
        'u': 24.07, 
        'g': 25.60, 
        'r': 25.81, 
        'i': 25.13, 
        'z': 24.13, 
        'y': 23.39
    }
    
    name = ''
    data_folder = os.path.join(folder, name)
    name_list = numpy.sort(os.listdir(data_folder))
    
    data = {
        'shear': numpy.array([]),
        'redshift': numpy.array([]),
        'mag_u_lsst': numpy.array([]),
        'mag_g_lsst': numpy.array([]),
        'mag_r_lsst': numpy.array([]),
        'mag_i_lsst': numpy.array([]),
        'mag_z_lsst': numpy.array([]),
        'mag_y_lsst': numpy.array([]), 
        'mag_err_u_lsst': numpy.array([]),
        'mag_err_g_lsst': numpy.array([]),
        'mag_err_r_lsst': numpy.array([]),
        'mag_err_i_lsst': numpy.array([]),
        'mag_err_z_lsst': numpy.array([]),
        'mag_err_y_lsst': numpy.array([]),
    }
    
    for name in name_list[:1]:
        print('Name: {}'.format(name))
        
        with h5py.File(os.path.join(data_folder, name), 'r') as file:
            key_list = numpy.sort([key for key in file.keys() if key != 'metaData'])
            for key in key_list:
                if list(file[key].keys()):
                    data['redshift'] = numpy.concatenate((data['redshift'], file[key]['redshift'][:].astype(numpy.float64)))
                    
                    data['mag_u_lsst'] = numpy.concatenate((data['mag_u_lsst'], file[key]['LSST_obs_u'][:].astype(numpy.float64)))
                    data['mag_err_u_lsst'] = numpy.power(10, 0.4 * (file[key]['LSST_obs_u'][:].astype(numpy.float64) - 24.07)) * 0.5 / numpy.log(10)
                    
                    data['mag_g_lsst'] = numpy.concatenate((data['mag_g_lsst'], file[key]['LSST_obs_g'][:].astype(numpy.float64)))
                    data['mag_err_g_lsst'] = numpy.power(10, 0.4 * (file[key]['LSST_obs_g'][:].astype(numpy.float64) - 25.60)) * 0.5 / numpy.log(10)
                    
                    data['mag_r_lsst'] = numpy.concatenate((data['mag_r_lsst'], file[key]['LSST_obs_r'][:].astype(numpy.float64)))
                    data['mag_err_r_lsst'] = numpy.power(10, 0.4 * (file[key]['LSST_obs_r'][:].astype(numpy.float64) - 25.81)) * 0.5 / numpy.log(10)
                    
                    data['mag_i_lsst'] = numpy.concatenate((data['mag_i_lsst'], file[key]['LSST_obs_i'][:].astype(numpy.float64)))
                    data['mag_err_i_lsst'] = numpy.power(10, 0.4 * (file[key]['LSST_obs_i'][:].astype(numpy.float64) - 25.13)) * 0.5 / numpy.log(10)
                    
                    data['mag_z_lsst'] = numpy.concatenate((data['mag_z_lsst'], file[key]['LSST_obs_z'][:].astype(numpy.float64)))
                    data['mag_err_z_lsst'] = numpy.power(10, 0.4 * (file[key]['LSST_obs_z'][:].astype(numpy.float64) - 24.13)) * 0.5 / numpy.log(10)
                    
                    data['mag_y_lsst'] = numpy.concatenate((data['mag_y_lsst'], file[key]['LSST_obs_y'][:].astype(numpy.float64)))
                    data['mag_err_y_lsst'] = numpy.power(10, 0.4 * (file[key]['LSST_obs_y'][:].astype(numpy.float64) - 23.39)) * 0.5 / numpy.log(10)
    
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
    PARSE.add_argument('--path', type=str, required=True, help='The base path of the project')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the simulation')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets for analysis')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the processes for multiprocessing')
    
    PATH = PARSE.parse_args().path
    FOLDER = PARSE.parse_args().folder
    LENGTH = PARSE.parse_args().length
    NUMBER = PARSE.parse_args().number
    
    # Output
    OUTPUT = main(PATH, FOLDER, LENGTH, NUMBER)