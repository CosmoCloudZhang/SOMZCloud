import os
import h5py
import time
import yaml
import numpy
import pandas
import argparse
from rail import core
from photerr import LsstErrorModel
from rail.estimation.algos import somoclu_som

def main(tag, index, folder):
    '''
    Create the application datasets
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of the application datasets
        folder (str): The base folder containing the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Path
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    som_folder = os.path.join(folder, 'SOM/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    os.makedirs(os.path.join(dataset_folder, 'CATALOG/'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/OBSERVATION/'.format(tag)), exist_ok=True)
    
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/APPLICATION/'.format(tag)), exist_ok=True)
    
    # Observation
    with open(os.path.join(dataset_folder, 'CATALOG/OBSERVE.yaml'), 'r') as file:
        observation_list = yaml.safe_load(file)['healpix_pixels']
    
    # Load
    observation_dataset = {
        'mu': numpy.array([]),
        'eta': numpy.array([]),
        'sigma': numpy.array([]),
        'major': numpy.array([]), 
        'minor': numpy.array([]),
        'redshift': numpy.array([]),
        'mag_u_lsst': numpy.array([]),
        'mag_g_lsst': numpy.array([]),
        'mag_r_lsst': numpy.array([]),
        'mag_i_lsst': numpy.array([]),
        'mag_z_lsst': numpy.array([]),
        'mag_y_lsst': numpy.array([]),
        'major_disk': numpy.array([]),
        'major_bulge': numpy.array([]), 
        'magnification': numpy.array([]),
        'ellipticity_disk': numpy.array([]),
        'ellipticity_bulge': numpy.array([]), 
        'bulge_to_total_ratio': numpy.array([])
    }
    
    for value in observation_list:
        
        with h5py.File(os.path.join(dataset_folder, '{}/OBSERVATION/OBSERVATION_{}.hdf5'.format(tag, value)), 'r') as file:
            
            observation_dataset['mu'] = numpy.append(observation_dataset['mu'], file['mu'][:].astype(numpy.float32), axis=0)
            observation_dataset['eta'] = numpy.append(observation_dataset['eta'], file['eta'][:].astype(numpy.float32), axis=0)
            observation_dataset['sigma'] = numpy.append(observation_dataset['sigma'], file['sigma'][:].astype(numpy.float32), axis=0)
            
            observation_dataset['major'] = numpy.append(observation_dataset['major'], file['major'][:].astype(numpy.float32), axis=0)
            observation_dataset['minor'] = numpy.append(observation_dataset['minor'], file['minor'][:].astype(numpy.float32), axis=0)
            
            observation_dataset['major_disk'] = numpy.append(observation_dataset['major_disk'], file['major_disk'][:].astype(numpy.float32), axis=0)
            observation_dataset['major_bulge'] = numpy.append(observation_dataset['major_bulge'], file['major_bulge'][:].astype(numpy.float32), axis=0)
            
            observation_dataset['ellipticity_disk'] = numpy.append(observation_dataset['ellipticity_disk'], file['ellipticity_disk'][:].astype(numpy.float32), axis=0)
            observation_dataset['ellipticity_bulge'] = numpy.append(observation_dataset['ellipticity_bulge'], file['ellipticity_bulge'][:].astype(numpy.float32), axis=0)
            
            observation_dataset['redshift'] = numpy.append(observation_dataset['redshift'], file['redshift'][:].astype(numpy.float32), axis=0)
            observation_dataset['magnification'] = numpy.append(observation_dataset['magnification'], file['magnification'][:].astype(numpy.float32), axis=0)
            observation_dataset['bulge_to_total_ratio'] = numpy.append(observation_dataset['bulge_to_total_ratio'], file['bulge_to_total_ratio'][:].astype(numpy.float32), axis=0)
            
            observation_dataset['mag_u_lsst'] = numpy.append(observation_dataset['mag_u_lsst'], file['mag_u_lsst'][:].astype(numpy.float32), axis=0)
            observation_dataset['mag_g_lsst'] = numpy.append(observation_dataset['mag_g_lsst'], file['mag_g_lsst'][:].astype(numpy.float32), axis=0)
            observation_dataset['mag_r_lsst'] = numpy.append(observation_dataset['mag_r_lsst'], file['mag_r_lsst'][:].astype(numpy.float32), axis=0)
            observation_dataset['mag_i_lsst'] = numpy.append(observation_dataset['mag_i_lsst'], file['mag_i_lsst'][:].astype(numpy.float32), axis=0)
            observation_dataset['mag_z_lsst'] = numpy.append(observation_dataset['mag_z_lsst'], file['mag_z_lsst'][:].astype(numpy.float32), axis=0)
            observation_dataset['mag_y_lsst'] = numpy.append(observation_dataset['mag_y_lsst'], file['mag_y_lsst'][:].astype(numpy.float32), axis=0)
    print(len(observation_dataset['redshift']))
    
    # Error
    error_model = LsstErrorModel(
        nYrObs=1, 
        sigLim=3.0,
        absFlux=True,
        ndMode='sigLim', 
        majorCol='major', 
        minorCol='minor', 
        decorrelate=True,
        extendedType='auto',
        renameDict={
            'u': 'mag_u_lsst',
            'g': 'mag_g_lsst',
            'r': 'mag_r_lsst',
            'i': 'mag_i_lsst',
            'z': 'mag_z_lsst',
            'y': 'mag_y_lsst'
        }
    )
    
    # Application
    application_dataset = error_model(pandas.DataFrame(observation_dataset), random_state=index)
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(som_folder, '{}/INFORM/INFORM.pkl'.format(tag))
    column_list = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    chunk = 100000
    application_size = len(application_dataset['redshift'])
    application_coordinate = numpy.zeros((application_size, 2), dtype=numpy.int32)
    
    for m in range(application_size // chunk + 1):
        begin = m * chunk
        stop = min((m + 1) * chunk, application_size)
        application = {key: application_dataset[key][begin: stop].astype(numpy.float32) for key in column_list}
        
        application_column = somoclu_som._computemagcolordata(data=application, mag_column_name='mag_i_lsst', column_names=column_list, colusage='colors')
        application_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], application_column)
    
    application_coordinate1 = application_coordinate[:, 0]
    application_coordinate2 = application_coordinate[:, 1]
    application_label = application_coordinate1 * model['n_columns'] + application_coordinate2
    application_occupation = numpy.bincount(application_label, minlength=model['n_rows'] * model['n_columns'])
    application_mean = numpy.divide(numpy.bincount(application_label, weights=application_dataset['redshift'], minlength=model['n_rows'] * model['n_columns']), application_occupation, out=numpy.ones(model['n_rows'] * model['n_columns']) * numpy.nan, where=application_occupation != 0)
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('meta')
        file['meta'].create_dataset('label', data=application_label, dtype=numpy.int32)
        file['meta'].create_dataset('coordinate1', data=application_coordinate1, dtype=numpy.int32)
        file['meta'].create_dataset('coordinate2', data=application_coordinate2, dtype=numpy.int32)
        
        file['meta'].create_dataset('mean', data=application_mean, dtype=numpy.float32)
        file['meta'].create_dataset('occupation', data=application_occupation, dtype=numpy.int32)
        
        file.create_group('photometry')
        file['photometry'].create_dataset('redshift', data=application_dataset['redshift'], dtype=numpy.float32)
        
        file['photometry'].create_dataset('mag_u_lsst', data=application_dataset['mag_u_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_g_lsst', data=application_dataset['mag_g_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_r_lsst', data=application_dataset['mag_r_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_i_lsst', data=application_dataset['mag_i_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_z_lsst', data=application_dataset['mag_z_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_y_lsst', data=application_dataset['mag_y_lsst'], dtype=numpy.float32)
        
        file['photometry'].create_dataset('mag_u_lsst_err', data=application_dataset['mag_u_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_g_lsst_err', data=application_dataset['mag_g_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_r_lsst_err', data=application_dataset['mag_r_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_i_lsst_err', data=application_dataset['mag_i_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_z_lsst_err', data=application_dataset['mag_z_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_y_lsst_err', data=application_dataset['mag_y_lsst_err'], dtype=numpy.float32)
        
        file.create_group('morphology')
        file['morphology'].create_dataset('mu', data=application_dataset['mu'], dtype=numpy.float32)
        file['morphology'].create_dataset('eta', data=application_dataset['eta'], dtype=numpy.float32)
        file['morphology'].create_dataset('sigma', data=application_dataset['sigma'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('major', data=application_dataset['major'], dtype=numpy.float32)
        file['morphology'].create_dataset('minor', data=application_dataset['minor'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('major_disk', data=application_dataset['major_disk'], dtype=numpy.float32)
        file['morphology'].create_dataset('major_bulge', data=application_dataset['major_bulge'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('ellipticity_disk', data=application_dataset['ellipticity_disk'], dtype=numpy.float32)
        file['morphology'].create_dataset('ellipticity_bulge', data=application_dataset['ellipticity_bulge'], dtype=numpy.float32)
        file['morphology'].create_dataset('bulge_to_total_ratio', data=application_dataset['bulge_to_total_ratio'], dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Application Datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the application datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the application datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)