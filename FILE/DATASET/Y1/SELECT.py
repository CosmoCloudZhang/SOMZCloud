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
    Create the selection datasets
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of the selection datasets
        folder (str): The base folder of the selection datasets datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Path
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    som_folder = os.path.join(folder, 'SOM/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/SELECTION/'.format(tag)), exist_ok=True)
    
    # Load
    with h5py.File(os.path.join(dataset_folder, '{}/SIMULATION/SIMULATION.hdf5'.format(tag)), 'r') as file:
        simulation_dataset = {key: file[key][:].astype(numpy.float32) for key in file.keys()}
    simulation_size = len(simulation_dataset['redshift'])
    
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
    
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_size = len(file['photometry']['redshift'])
    
    indices = numpy.random.choice(simulation_size, size=application_size, replace=False)
    selection_dataset = dict(error_model(pandas.DataFrame(simulation_dataset).iloc[indices], random_state=index))
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(som_folder, '{}/INFORM/INFORM.pkl'.format(tag))
    column_list = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    chunk = 100000
    selection_size = len(selection_dataset['redshift'])
    selection_coordinate = numpy.zeros((selection_size, 2), dtype=numpy.int32)
    
    for m in range(selection_size // chunk + 1):
        begin = m * chunk
        stop = min((m + 1) * chunk, selection_size)
        selection = {key: selection_dataset[key][begin: stop].astype(numpy.float32) for key in column_list}
        
        selection_column = somoclu_som._computemagcolordata(data=selection, mag_column_name='mag_i_lsst', column_names=column_list, colusage='magandcolors')
        selection_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], selection_column)
    
    selection_coordinate1 = selection_coordinate[:, 0]
    selection_coordinate2 = selection_coordinate[:, 1]
    selection_label = selection_coordinate1 * model['n_columns'] + selection_coordinate2
    
    som_size = model['n_rows'] * model['n_columns']
    selection_count = numpy.bincount(selection_label, minlength=som_size)
    selection_mean = numpy.divide(numpy.bincount(selection_label, weights=selection_dataset['redshift'], minlength=som_size), selection_count, out=numpy.ones(som_size) * numpy.nan, where=selection_count != 0)
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/SELECTION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('meta')
        
        file['meta'].create_dataset('mean', data=selection_mean, dtype=numpy.float32)
        file['meta'].create_dataset('count', data=selection_count, dtype=numpy.int32)
        
        file['meta'].create_dataset('label', data=selection_label, dtype=numpy.int32)
        file['meta'].create_dataset('coordinate1', data=selection_coordinate1, dtype=numpy.int32)
        file['meta'].create_dataset('coordinate2', data=selection_coordinate2, dtype=numpy.int32)
        
        file.create_group('photometry')
        file['photometry'].create_dataset('redshift', data=selection_dataset['redshift'], dtype=numpy.float32)
        file['photometry'].create_dataset('redshift_true', data=selection_dataset['redshift_true'], dtype=numpy.float32)
        
        file['photometry'].create_dataset('mag_u_lsst', data=selection_dataset['mag_u_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_g_lsst', data=selection_dataset['mag_g_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_r_lsst', data=selection_dataset['mag_r_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_i_lsst', data=selection_dataset['mag_i_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_z_lsst', data=selection_dataset['mag_z_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_y_lsst', data=selection_dataset['mag_y_lsst'], dtype=numpy.float32)
        
        file['photometry'].create_dataset('mag_u_lsst_err', data=selection_dataset['mag_u_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_g_lsst_err', data=selection_dataset['mag_g_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_r_lsst_err', data=selection_dataset['mag_r_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_i_lsst_err', data=selection_dataset['mag_i_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_z_lsst_err', data=selection_dataset['mag_z_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_y_lsst_err', data=selection_dataset['mag_y_lsst_err'], dtype=numpy.float32)
        
        file.create_group('morphology')
        file['morphology'].create_dataset('value', data=selection_dataset['value'], dtype=numpy.float32)
        file['morphology'].create_dataset('galaxy_id', data=selection_dataset['galaxy_id'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('mu', data=selection_dataset['mu'], dtype=numpy.float32)
        file['morphology'].create_dataset('eta', data=selection_dataset['eta'], dtype=numpy.float32)
        file['morphology'].create_dataset('sigma', data=selection_dataset['sigma'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('major', data=selection_dataset['major'], dtype=numpy.float32)
        file['morphology'].create_dataset('minor', data=selection_dataset['minor'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('major_disk', data=selection_dataset['major_disk'], dtype=numpy.float32)
        file['morphology'].create_dataset('major_bulge', data=selection_dataset['major_bulge'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('ellipticity_disk', data=selection_dataset['ellipticity_disk'], dtype=numpy.float32)
        file['morphology'].create_dataset('ellipticity_bulge', data=selection_dataset['ellipticity_bulge'], dtype=numpy.float32)
        file['morphology'].create_dataset('bulge_to_total_ratio', data=selection_dataset['bulge_to_total_ratio'], dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Selection Datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the selection datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the selection datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    index = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, index, FOLDER)