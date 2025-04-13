import os
import time
import h5py
import numpy
import scipy
import argparse
from rail import core
from sklearn import cluster


def main(tag, index, folder):
    '''
    ensemble summarization of the source samples
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    numpy.random.seed(index)
    print('Index: {}'.format(index))
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    
    os.makedirs(os.path.join(summarize_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}'.format(tag, index)), exist_ok=True)
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(dataset_folder, '{}/SOM/INFORM.pkl'.format(tag))
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size = file['meta']['cell_size'][...]
        application_cell_id = file['meta']['cell_id'][...]
        application_sigma = file['morphology']['sigma'][...]
    
    # Select
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_source = file['bin_source'][...]
        application_z_phot = file['z_phot'][...]
    
    with h5py.File(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        select_source = file['select'][...]
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size = file['meta']['cell_size'][...]
        combination_cell_id = file['meta']['cell_id'][...]
        combination_redshift = file['photometry']['redshift'][...]
    
    # Reference
    with h5py.File(os.path.join(model_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_source = file['bin_source'][...]
        combination_z_phot = file['z_phot'][...]
    
    with h5py.File(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/REFERENCE.hdf5'.format(tag, index)), 'r') as file:
        reference_source = file['reference'][...]
    
    # Size
    data_size = 100
    bin_source_size = len(bin_source) - 1
    data_source = numpy.zeros((bin_source_size, data_size, grid_size + 1))
    
    # Cluster
    som_model = model['som']
    cluster_size = cell_size // 2
    
    som_model.cluster(cluster.AgglomerativeClustering(n_clusters=cluster_size, linkage='complete'))
    cluster_id = som_model.clusters.flatten()
    
    # Reference
    reference = numpy.sum(reference_source, axis=0) > 0
    reference_size = numpy.sum(reference)
    
    # Combination
    combination_z_phot_reference = combination_z_phot[reference]
    combination_z_spec_reference = combination_redshift[reference]
    combination_cell_id_reference = combination_cell_id[reference]
    
    # Loop
    for m in range(bin_source_size):
        # Select
        select = select_source[m, :] 
        select_size = numpy.sum(select)
        
        # Application
        application_sigma_select = application_sigma[select]
        application_z_phot_select = application_z_phot[select]
        application_cell_id_select = application_cell_id[select]
        
        # Bootstrap
        for n in range(data_size):
            # Application
            application_indices = numpy.random.choice(numpy.arange(select_size), size=select_size, replace=True)
            
            application_sigma_data = application_sigma_select[application_indices]
            application_z_phot_data = application_z_phot_select[application_indices]
            application_cell_id_data = application_cell_id_select[application_indices]
            
            application_cluster_id_data = cluster_id[application_cell_id_data]
            application_cluster_count_data = numpy.bincount(application_cluster_id_data, weights=1 / numpy.square(application_sigma_data), minlength=cluster_size)
            
            application_cluster_z_phot_data = numpy.bincount(application_cluster_id_data, weights=application_z_phot_data / numpy.square(application_sigma_data), minlength=cluster_size)
            application_cluster_z_phot_data = numpy.divide(application_cluster_z_phot_data, application_cluster_count_data, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=application_cluster_count_data > 0)
            
            # Combination
            combination_indices = numpy.random.choice(numpy.arange(reference_size), size=reference_size, replace=True)
            
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
            
            factor = scipy.integrate.trapezoid(x=z_grid, y=ensemble_cluster, axis=1)
            ensemble_cluster = numpy.divide(ensemble_cluster, factor[:, numpy.newaxis], out=numpy.zeros((cluster_size, grid_size + 1)), where=factor[:, numpy.newaxis] > 0)
            
            # Ensemble
            ensemble = numpy.average(ensemble_cluster, axis=0, weights=application_cluster_count_data)
            data_source[m, n, :] = ensemble / scipy.integrate.trapezoid(x=z_grid, y=ensemble, axis=0)
    
    # Average
    average_source = numpy.mean(data_source, axis=1)
    average_source = average_source / scipy.integrate.trapezoid(x=z_grid, y=average_source, axis=1)[:, numpy.newaxis]
    
    # Save
    with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/HISTOGRAM.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('data', data=data_source, dtype=numpy.float32)
        file.create_dataset('average', data=average_source, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Summarize Histogram')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)
