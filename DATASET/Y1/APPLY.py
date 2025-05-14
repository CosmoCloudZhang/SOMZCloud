import os
import h5py
import time
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
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/APPLICATION/'.format(tag)), exist_ok=True)
    
    # Load
    with h5py.File(os.path.join(dataset_folder, '{}/OBSERVATION/OBSERVATION.hdf5'.format(tag)), 'r') as file:
        application_dataset = {key: file[key][...] for key in file.keys()}
    
    # Error
    error_model = LsstErrorModel(
        sigLim=1.0,
        absFlux=True,
        ndMode='sigLim', 
        majorCol='major', 
        minorCol='minor', 
        decorrelate=True,
        extendedType='auto',
        nYrObs=int(tag[1:]), 
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
    application_dataset = error_model(pandas.DataFrame(application_dataset), random_state=index)
    
    # Filter
    snr = 15    
    magnitude1 = 16
    magnitude2 = error_model.getLimitingMags(nSigma=snr, coadded=True, aperture=0)['mag_i_lsst']
    filter = (magnitude1 < application_dataset['mag_i_lsst'].values) & (application_dataset['mag_i_lsst'].values < magnitude2)
    
    random_generator = numpy.random.default_rng(seed=index)
    value_list = numpy.unique(application_dataset['value'].values)
    filter = filter & (numpy.isin(application_dataset['value'].values, random_generator.choice(value_list, len(value_list) // 2, replace=False)))
    
    # Inform
    application_dataset = {key: application_dataset[key].values[filter] for key in application_dataset.keys()}
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(dataset_folder, '{}/SOM/INFORM.pkl'.format(tag))
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    chunk = 100000
    application_size = len(application_dataset['redshift'])
    application_cell_coordinate = numpy.zeros((application_size, 2), dtype=numpy.int32)
    
    for m in range(application_size // chunk + 1):
        begin = m * chunk
        stop = min((m + 1) * chunk, application_size)
        
        if begin < stop:
            application = {key: application_dataset[key][begin: stop] for key in model['usecols']}
            
            application_column = somoclu_som._computemagcolordata(data=application, mag_column_name=model['ref_column'], column_names=model['usecols'], colusage=model['column_usage'])
            application_cell_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], application_column)
    
    application_cell_coordinate1 = application_cell_coordinate[:, 0]
    application_cell_coordinate2 = application_cell_coordinate[:, 1]
    application_cell_id = numpy.ravel_multi_index(numpy.transpose(application_cell_coordinate), (model['n_rows'], model['n_columns']))
    
    cell_size = model['n_rows'] * model['n_columns']
    application_cell_count = numpy.bincount(application_cell_id, minlength=cell_size)
    application_cell_z_true = numpy.divide(numpy.bincount(application_cell_id, weights=application_dataset['redshift_true'], minlength=cell_size), application_cell_count, out=numpy.ones(cell_size) * numpy.nan, where=application_cell_count != 0)
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('meta')
        file['meta'].create_dataset('cell_size', data=cell_size, dtype=numpy.int32)
        file['meta'].create_dataset('cell_size1', data=model['n_rows'], dtype=numpy.int32)
        file['meta'].create_dataset('cell_size2', data=model['n_columns'], dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_id', data=application_cell_id, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate1', data=application_cell_coordinate1, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate2', data=application_cell_coordinate2, dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_count', data=application_cell_count, dtype=numpy.int32)
        file['meta'].create_dataset('cell_z_true', data=application_cell_z_true, dtype=numpy.float32)
        
        file.create_group('photometry')
        file['photometry'].create_dataset('redshift', data=application_dataset['redshift'], dtype=numpy.float32)
        file['photometry'].create_dataset('redshift_true', data=application_dataset['redshift_true'], dtype=numpy.float32)
        
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
        file['morphology'].create_dataset('ra', data=application_dataset['ra'], dtype=numpy.float32)
        file['morphology'].create_dataset('dec', data=application_dataset['dec'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('id', data=application_dataset['id'], dtype=numpy.int32)
        file['morphology'].create_dataset('value', data=application_dataset['value'], dtype=numpy.int32)
        
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