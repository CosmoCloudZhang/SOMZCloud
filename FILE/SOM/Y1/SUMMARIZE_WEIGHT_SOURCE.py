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
    Histogram of the spectroscopic redshifts of the source samples
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    som_folder = os.path.join(folder, 'SOM/')
    fzb_folder = os.path.join(folder, 'FZB/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    os.makedirs(os.path.join(som_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(som_folder, '{}/SOURCE/SOURCE{}'.format(tag, index)), exist_ok=True)
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(som_folder, '{}/INFORM/INFORM.pkl'.format(tag))
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_delta = (z2 - z1) / grid_size
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    z_bin = numpy.linspace(z1 - z_delta / 2, z2 + z_delta / 2, z_grid.size + 1)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_cell_id = file['meta']['cell_id'][...]
    
    # Select
    with h5py.File(os.path.join(fzb_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_source = file['bin_source'][...]
        application_z_phot = file['z_phot'][...]
    
    with h5py.File(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        select_source = file['select'][...]
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_cell_id = file['meta']['cell_id'][...]
        combination_redshift = file['photometry']['redshift'][...]
    
    # Reference
    with h5py.File(os.path.join(fzb_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_source = file['bin_source'][...]
        combination_z_phot = file['z_phot'][...]
    
    with h5py.File(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}/REFERENCE.hdf5'.format(tag, index)), 'r') as file:
        reference_source = file['reference'][...]
    
    # Size
    sample_size = 100
    bin_source_size = len(bin_source) - 1
    
    # Lens
    single_source = numpy.zeros((bin_source_size, grid_size + 1))
    sample_source = numpy.zeros((bin_source_size, sample_size, grid_size + 1))
    
    # Cluster
    som_model = model['som']
    cluster_size = model['n_rows'] * model['n_columns'] // 4
    
    som_model.cluster(cluster.AgglomerativeClustering(n_clusters=cluster_size, linkage='complete'))
    cluster_id = som_model.clusters.flatten()
    
    # Loop
    for m in range(bin_source_size):
        # Select
        select = select_source[m, :] 
        select_size = numpy.sum(select)
        
        # Reference
        reference = reference_source[m, :]
        reference_size = numpy.sum(reference)
        
        # Application
        application_z_phot_select = application_z_phot[select]
        application_cell_id_select = application_cell_id[select]
        
        application_cluster_id_select = cluster_id[application_cell_id_select]
        application_cluster_count_select = numpy.bincount(application_cluster_id_select, minlength=cluster_size)
        
        application_cluster_z_phot_select = numpy.divide(numpy.bincount(application_cluster_id_select, weights=application_z_phot_select, minlength=cluster_size), application_cluster_count_select, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=application_cluster_count_select > 0)
        
        # Combination
        combination_z_phot_reference = combination_z_phot[reference]
        combination_z_spec_reference = combination_redshift[reference]
        
        combination_cell_id_reference = combination_cell_id[reference]
        combination_cluster_id_reference = cluster_id[combination_cell_id_reference]
        combination_cluster_count_reference = numpy.bincount(combination_cluster_id_reference, minlength=cluster_size)
        
        combination_cluster_z_phot_reference = numpy.divide(numpy.bincount(combination_cluster_id_reference, weights=combination_z_phot_reference, minlength=cluster_size), combination_cluster_count_reference, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=combination_cluster_count_reference > 0)
        combination_cluster_z_spec_reference = numpy.divide(numpy.bincount(combination_cluster_id_reference, weights=combination_z_spec_reference, minlength=cluster_size), combination_cluster_count_reference, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=combination_cluster_count_reference > 0)
        
        # Filter
        filter = combination_cluster_count_reference > 0
        cluster_mean_delta = application_cluster_z_phot_select - combination_cluster_z_spec_reference
        sigma = 1.4826 * numpy.median(numpy.abs(cluster_mean_delta[filter] - numpy.median(cluster_mean_delta[filter])))
        filter = filter & (numpy.abs(cluster_mean_delta) - 5 * sigma < 0.00) & (numpy.abs(combination_cluster_z_phot_reference - combination_cluster_z_spec_reference) < 0.02)
        
        # Weight
        cluster_weight = numpy.divide(application_cluster_count_select, combination_cluster_count_reference, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=filter)
        weight = cluster_weight[combination_cluster_id_reference]
        
        # Single
        histogram = numpy.histogram(combination_z_spec_reference, bins=z_bin, weights=weight, range=(z1, z2), density=True)[0]
        single_source[m, :] = histogram / scipy.integrate.trapezoid(x=z_grid, y=histogram, axis=0)
        
        # Bootstrap
        for n in range(sample_size):
            # Application
            application_indices = numpy.random.choice(numpy.arange(select_size), size=select_size, replace=True)
            
            application_z_phot_sample = application_z_phot_select[application_indices]
            application_cell_id_sample = application_cell_id_select[application_indices]
            
            application_cluster_id_sample = cluster_id[application_cell_id_sample]
            application_cluster_count_sample = numpy.bincount(application_cluster_id_sample, minlength=cluster_size)
            
            application_cluster_z_phot_sample = numpy.divide(numpy.bincount(application_cluster_id_sample, weights=application_z_phot_sample, minlength=cluster_size), application_cluster_count_sample, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=application_cluster_count_sample > 0)
            
            # Combination
            combination_indices = numpy.random.choice(numpy.arange(reference_size), size=reference_size, replace=True)
            
            combination_z_phot_sample = combination_z_phot_reference[combination_indices]
            combination_z_spec_sample = combination_z_spec_reference[combination_indices]
            
            combination_cell_id_sample = combination_cell_id_reference[combination_indices]
            combination_cluster_id_sample = cluster_id[combination_cell_id_sample]
            combination_cluster_count_sample = numpy.bincount(combination_cluster_id_sample, minlength=cluster_size)
            
            combination_cluster_z_phot_sample = numpy.divide(numpy.bincount(combination_cluster_id_sample, weights=combination_z_phot_sample, minlength=cluster_size), combination_cluster_count_sample, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=combination_cluster_count_sample > 0)
            combination_cluster_z_spec_sample = numpy.divide(numpy.bincount(combination_cluster_id_sample, weights=combination_z_spec_sample, minlength=cluster_size), combination_cluster_count_sample, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=combination_cluster_count_sample > 0)
            
            # Filter
            filter_sample = combination_cluster_count_sample > 0
            cluster_mean_delta_sample = application_cluster_z_phot_sample - combination_cluster_z_spec_sample
            sigma_sample = 1.4826 * numpy.median(numpy.abs(cluster_mean_delta_sample[filter_sample] - numpy.median(cluster_mean_delta_sample[filter_sample])))
            filter_sample = filter_sample & (numpy.abs(cluster_mean_delta_sample) - 5 * sigma_sample < 0.00) & (numpy.abs(combination_cluster_z_phot_sample - combination_cluster_z_spec_sample) < 0.02)
            
            # Weight
            cluster_weight_sample = numpy.divide(application_cluster_count_sample, combination_cluster_count_sample, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=filter_sample)
            weight_sample = cluster_weight_sample[combination_cluster_id_sample]
            
            # Sample
            histogram = numpy.histogram(combination_z_spec_sample, bins=z_bin, weights=weight_sample, range=(z1, z2), density=True)[0]
            sample_source[m, n, :] = histogram / scipy.integrate.trapezoid(x=z_grid, y=histogram, axis=0)
    
    # Save
    with h5py.File(os.path.join(som_folder, '{}/SOURCE/SOURCE{}/SUMMARIZE_WEIGHT.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('single', data=single_source, dtype=numpy.float32)
        file.create_dataset('sample', data=sample_source, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Summarize Weight')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)
