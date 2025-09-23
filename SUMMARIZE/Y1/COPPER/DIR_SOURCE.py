import os
import time
import h5py
import numpy
import scipy
import argparse
from rail import core
from sklearn import cluster


def main(tag, name, index, folder):
    '''
    DIR summarization of the source samples
    
    Arguments:
        tag (str): The tag of configuration
        name (str): The name of configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    print('Index: {}'.format(index))
    random_generator = numpy.random.default_rng(index)
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    os.makedirs(os.path.join(summarize_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(summarize_folder, '{}/{}/SOURCE/'.format(tag, name)), exist_ok=True)
    os.makedirs(os.path.join(summarize_folder, '{}/{}/SOURCE/SOURCE{}'.format(tag, name, index)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Application
    with h5py.File(os.path.join(summarize_folder, '{}/{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, name, index)), 'r') as file:
        cell_size = file['application']['cell_size'][...]
        application_cell_id = file['application']['cell_id'][...]
    
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_sigma = file['morphology']['sigma'][...]
    
    # Target
    with h5py.File(os.path.join(model_folder, '{}/TARGET/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_source = file['bin_source'][...]
        application_z_phot = file['z_phot'][...]
    
    with h5py.File(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/TARGET.hdf5'.format(tag, index)), 'r') as file:
        target_source = file['target'][...]
    
    # Combination
    with h5py.File(os.path.join(summarize_folder, '{}/{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, name, index)), 'r') as file:
        cell_size = file['combination']['cell_size'][...]
        combination_cell_id = file['combination']['cell_id'][...]
    
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_redshift = file['photometry']['redshift'][...]
    
    # Reference
    with h5py.File(os.path.join(model_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_source = file['bin_source'][...]
        combination_z_phot = file['z_phot'][...]
    
    with h5py.File(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/REFERENCE.hdf5'.format(tag, index)), 'r') as file:
        reference_source = file['reference'][...]
    
    reference = numpy.any(reference_source, axis=0)
    reference_size = numpy.sum(reference)
    
    combination_z_phot_reference = combination_z_phot[reference]
    combination_z_spec_reference = combination_redshift[reference]
    combination_cell_id_reference = combination_cell_id[reference]
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(summarize_folder, '{}/{}/INFORM/INFORM{}.pkl'.format(tag, name, index))
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    # Cluster
    som_model = model['som']
    cluster_size = cell_size // 2
    
    som_model.cluster(cluster.AgglomerativeClustering(n_clusters=cluster_size, linkage='complete'))
    cluster_id = som_model.clusters.flatten()
    
    # Size
    data_size = 100
    bin_source_size = len(bin_source) - 1
    data_source = numpy.zeros((bin_source_size, data_size, grid_size + 1))
    
    # Loop
    for m in range(bin_source_size):
        # Target
        target = target_source[m, :] 
        target_size = numpy.sum(target)
        
        # Application
        application_sigma_target = application_sigma[target]
        application_z_phot_target = application_z_phot[target]
        application_cell_id_target = application_cell_id[target]
        
        if target_size > 0 and reference_size > 0:
            # Bootstrap
            for n in range(data_size):
                # Application
                application_indices = random_generator.choice(numpy.arange(target_size), size=target_size, replace=True)
                
                application_sigma_data = application_sigma_target[application_indices]
                application_z_phot_data = application_z_phot_target[application_indices]
                application_cell_id_data = application_cell_id_target[application_indices]
                
                application_cluster_id_data = cluster_id[application_cell_id_data]
                application_cluster_count_data = numpy.bincount(application_cluster_id_data, weights=1 / numpy.square(application_sigma_data), minlength=cluster_size)
                
                application_cluster_z_phot_data = numpy.bincount(application_cluster_id_data, weights=application_z_phot_data / numpy.square(application_sigma_data), minlength=cluster_size)
                application_cluster_z_phot_data = numpy.divide(application_cluster_z_phot_data, application_cluster_count_data, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=application_cluster_count_data > 0)
                
                # Combination
                combination_indices = random_generator.choice(numpy.arange(reference_size), size=reference_size, replace=True)
                
                combination_z_phot_data = combination_z_phot_reference[combination_indices]
                combination_z_spec_data = combination_z_spec_reference[combination_indices]
                combination_cell_id_data = combination_cell_id_reference[combination_indices]
                
                combination_cluster_id_data = cluster_id[combination_cell_id_data]
                combination_cluster_count_data = numpy.bincount(combination_cluster_id_data, minlength=cluster_size)
                
                combination_cluster_z_phot_data = numpy.bincount(combination_cluster_id_data, weights=combination_z_phot_data, minlength=cluster_size)
                combination_cluster_z_phot_data = numpy.divide(combination_cluster_z_phot_data, combination_cluster_count_data, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=combination_cluster_count_data > 0)
                
                combination_cluster_z_spec_data = numpy.bincount(combination_cluster_id_data, weights=combination_z_spec_data, minlength=cluster_size)
                combination_cluster_z_spec_data = numpy.divide(combination_cluster_z_spec_data, combination_cluster_count_data, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=combination_cluster_count_data > 0)
                
                # Filter
                filter_data = (application_cluster_count_data > 0) & (combination_cluster_count_data > 0)
                
                # Combination Mask
                combination_cluster_mask = filter_data[combination_cluster_id_data]
                combination_weight_data = numpy.array(combination_cluster_mask, dtype=numpy.float32)
                combination_ensemble_indices = numpy.digitize(combination_z_spec_data, bins=z_grid, right=False) - 1
                
                # Ensemble Cluster
                ensemble_cluster = numpy.zeros((cluster_size, grid_size + 1))
                numpy.add.at(ensemble_cluster, (combination_cluster_id_data[combination_cluster_mask], combination_ensemble_indices[combination_cluster_mask]), combination_weight_data[combination_cluster_mask])
                
                factor_cluster = scipy.integrate.trapezoid(x=z_grid, y=ensemble_cluster, axis=1)[:, numpy.newaxis]
                ensemble_cluster = numpy.divide(ensemble_cluster, factor_cluster, out=numpy.zeros((cluster_size, grid_size + 1)), where=factor_cluster > 0)
                
                # Ensemble
                ensemble = numpy.average(ensemble_cluster, axis=0, weights=application_cluster_count_data)
                
                data_factor = scipy.integrate.trapezoid(x=z_grid, y=ensemble, axis=0)
                data_source[m, n, :] = numpy.divide(ensemble, data_factor, out=numpy.zeros((grid_size + 1)), where=data_factor > 0)
        else:
            data_source[m, :, :] = numpy.zeros((data_size, grid_size + 1))
    
    # Average
    average_source = numpy.mean(data_source, axis=1)
    average_factor = scipy.integrate.trapezoid(x=z_grid, y=average_source, axis=1)[:, numpy.newaxis]
    average_source = numpy.divide(average_source, average_factor, out=numpy.zeros((bin_source_size, grid_size + 1)), where=average_factor > 0)
    
    # Save
    with h5py.File(os.path.join(summarize_folder, '{}/{}/SOURCE/SOURCE{}/DIR.hdf5'.format(tag, name, index)), 'w') as file:
        file.create_group('meta')
        file['meta'].create_dataset('z1', data=z1, dtype=numpy.float32)
        file['meta'].create_dataset('z2', data=z2, dtype=numpy.float32)
        file['meta'].create_dataset('z_grid', data=z_grid, dtype=numpy.float32)
        file['meta'].create_dataset('grid_size', data=grid_size, dtype=numpy.int32)
        file['meta'].create_dataset('data_size', data=data_size, dtype=numpy.int32)
        
        file['meta'].create_dataset('bin', data=bin_source, dtype=numpy.float32)
        file['meta'].create_dataset('bin_size', data=bin_source_size, dtype=numpy.int32)
        
        file.create_group('ensemble')
        file['ensemble'].create_dataset('data', data=data_source, dtype=numpy.float32)
        file['ensemble'].create_dataset('average', data=average_source, dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Summarize Iron DIR Source')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, INDEX, FOLDER)
