import os
import h5py
import time
import numpy
import argparse
from rail import core
from rail.estimation.algos import somoclu_som


def main(tag, index, folder):
    '''
    Create the association datasets
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of the association datasets
        folder (str): The base folder of the association datasets datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    random_generator = numpy.random.default_rng(seed=index)
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/ASSOCIATION/'.format(tag)), exist_ok=True)
    
    # Load
    with h5py.File(os.path.join(dataset_folder, '{}/SIMULATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        simulation_dataset = {key: file[key][...] for key in file.keys()}
    
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_cell_count = file['meta']['cell_count'][...]
        application_size = len(file['photometry']['redshift'][...])
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(dataset_folder, '{}/SOM/INFORM.pkl'.format(tag))
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    chunk = 100000
    simulation_size = len(simulation_dataset['redshift'])
    simulation_cell_coordinate = numpy.zeros((simulation_size, 2), dtype=numpy.int32)
    
    for m in range(simulation_size // chunk + 1):
        begin = m * chunk
        stop = min((m + 1) * chunk, simulation_size)
        
        if begin < stop:
            simulation = {key: simulation_dataset[key][begin: stop] for key in model['usecols']}
            
            simulation_column = somoclu_som._computemagcolordata(data=simulation, mag_column_name=model['ref_column'], column_names=model['usecols'], colusage=model['column_usage'])
            simulation_cell_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], simulation_column)
    
    simulation_cell_coordinate1 = simulation_cell_coordinate[:, 0]
    simulation_cell_coordinate2 = simulation_cell_coordinate[:, 1]
    simulation_cell_id = numpy.ravel_multi_index(numpy.transpose(simulation_cell_coordinate), (model['n_rows'], model['n_columns']))
    
    cell_size = model['n_rows'] * model['n_columns']
    simulation_cell_count = numpy.bincount(simulation_cell_id, minlength=cell_size)
    
    simulation_weight = numpy.divide(application_cell_count, simulation_cell_count, out=numpy.zeros(cell_size), where=simulation_cell_count != 0)
    simulation_probability = simulation_weight[simulation_cell_id] / numpy.sum(simulation_weight[simulation_cell_id])
    
    # Association
    indices = random_generator.choice(simulation_size, size=application_size, replace=True, p=simulation_probability)
    association_dataset = {key: simulation_dataset[key][indices] for key in simulation_dataset.keys()}
    
    association_cell_coordinate1 = simulation_cell_coordinate1[indices]
    association_cell_coordinate2 = simulation_cell_coordinate2[indices]
    association_cell_id = simulation_cell_id[indices]
    
    association_cell_count = numpy.bincount(association_cell_id, minlength=cell_size)
    association_cell_z_true = numpy.divide(numpy.bincount(association_cell_id, weights=association_dataset['redshift_true'], minlength=cell_size), association_cell_count, out=numpy.ones(cell_size) * numpy.nan, where=association_cell_count != 0)
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/ASSOCIATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('meta')
        
        file['meta'].create_dataset('cell_size', data=cell_size, dtype=numpy.int32)
        file['meta'].create_dataset('cell_size1', data=model['n_rows'], dtype=numpy.int32)
        file['meta'].create_dataset('cell_size2', data=model['n_columns'], dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_id', data=association_cell_id, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate1', data=association_cell_coordinate1, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate2', data=association_cell_coordinate2, dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_count', data=association_cell_count, dtype=numpy.int32)
        file['meta'].create_dataset('cell_z_true', data=association_cell_z_true, dtype=numpy.float32)
        
        file.create_group('photometry')
        file['photometry'].create_dataset('redshift', data=association_dataset['redshift'], dtype=numpy.float32)
        file['photometry'].create_dataset('redshift_true', data=association_dataset['redshift_true'], dtype=numpy.float32)
        
        file['photometry'].create_dataset('mag_u_lsst', data=association_dataset['mag_u_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_g_lsst', data=association_dataset['mag_g_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_r_lsst', data=association_dataset['mag_r_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_i_lsst', data=association_dataset['mag_i_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_z_lsst', data=association_dataset['mag_z_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_y_lsst', data=association_dataset['mag_y_lsst'], dtype=numpy.float32)
        
        file['photometry'].create_dataset('mag_u_lsst_err', data=association_dataset['mag_u_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_g_lsst_err', data=association_dataset['mag_g_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_r_lsst_err', data=association_dataset['mag_r_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_i_lsst_err', data=association_dataset['mag_i_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_z_lsst_err', data=association_dataset['mag_z_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_y_lsst_err', data=association_dataset['mag_y_lsst_err'], dtype=numpy.float32)
        
        file.create_group('morphology')
        file['morphology'].create_dataset('ra', data=association_dataset['ra'], dtype=numpy.float32)
        file['morphology'].create_dataset('dec', data=association_dataset['dec'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('id', data=association_dataset['id'], dtype=numpy.int32)
        file['morphology'].create_dataset('value', data=association_dataset['value'], dtype=numpy.int32)
        
        file['morphology'].create_dataset('mu', data=association_dataset['mu'], dtype=numpy.float32)
        file['morphology'].create_dataset('eta', data=association_dataset['eta'], dtype=numpy.float32)
        file['morphology'].create_dataset('sigma', data=association_dataset['sigma'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('major', data=association_dataset['major'], dtype=numpy.float32)
        file['morphology'].create_dataset('minor', data=association_dataset['minor'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('major_disk', data=association_dataset['major_disk'], dtype=numpy.float32)
        file['morphology'].create_dataset('major_bulge', data=association_dataset['major_bulge'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('radius', data=association_dataset['radius'], dtype=numpy.float32)
        file['morphology'].create_dataset('ellipticity_disk', data=association_dataset['ellipticity_disk'], dtype=numpy.float32)
        file['morphology'].create_dataset('ellipticity_bulge', data=association_dataset['ellipticity_bulge'], dtype=numpy.float32)
        file['morphology'].create_dataset('bulge_to_total_ratio', data=association_dataset['bulge_to_total_ratio'], dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Association Datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the association datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the association datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    index = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, index, FOLDER)