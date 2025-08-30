import os
import time
import h5py
import numpy
import scipy
import argparse
from rail import core
from sklearn import cluster


def main(tag, label, index, folder):
    '''
    Truth of the spectroscopic redshifts of the lens samples
    
    Arguments:
        tag (str): The tag of configuration
        label (str): The label of configuration
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
    constrain_folder = os.path.join(folder, 'CONSTRAIN/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    
    os.makedirs(os.path.join(summarize_folder, '{}/{}/LENS/'.format(label, tag)), exist_ok=True)
    os.makedirs(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}'.format(label, tag, index)), exist_ok=True)
    
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
        application_redshift_true = file['photometry']['redshift_true'][...]
    
    # Select
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_lens = file['bin_lens'][...]
        application_z_phot = file['z_phot'][...]
    
    with h5py.File(os.path.join(model_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        select_lens = file['select'][...]
    
    # Restriction
    with h5py.File(os.path.join(dataset_folder, '{}/RESTRICTION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size = file['meta']['cell_size'][...]
        restriction_cell_id = file['meta']['cell_id'][...]
        restriction_redshift = file['photometry']['redshift'][...]
    
    # Reference
    with h5py.File(os.path.join(constrain_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_lens = file['bin_lens'][...]
        restriction_z_phot = file['z_phot'][...]
    
    with h5py.File(os.path.join(constrain_folder, '{}/LENS/LENS{}/REFERENCE.hdf5'.format(tag, index)), 'r') as file:
        reference_lens = file['reference'][...]
    
    # Size
    data_size = 100
    bin_lens_size = len(bin_lens) - 1
    data_lens = numpy.zeros((bin_lens_size, data_size, grid_size + 1))
    
    # Cluster
    som_model = model['som']
    cluster_size = cell_size // 4
    
    som_model.cluster(cluster.AgglomerativeClustering(n_clusters=cluster_size, linkage='complete'))
    cluster_id = som_model.clusters.flatten()
    
    # Loop
    for m in range(bin_lens_size):
        # Select
        select = select_lens[m, :]
        select_size = numpy.sum(select)
        
        # Application
        application_z_phot_select = application_z_phot[select]
        application_cell_id_select = application_cell_id[select]
        application_redshift_true_select = application_redshift_true[select]
        
        # Reference
        reference = reference_lens[m, :]
        reference_size = numpy.sum(reference)
        
        # Restriction
        restriction_z_phot_reference = restriction_z_phot[reference]
        restriction_z_spec_reference = restriction_redshift[reference]
        restriction_cell_id_reference = restriction_cell_id[reference]
        
        # Bootstrap
        for n in range(data_size):
            # Application
            application_indices = random_generator.choice(numpy.arange(select_size), size=select_size, replace=True)
            
            application_z_phot_data = application_z_phot_select[application_indices]
            application_cell_id_data = application_cell_id_select[application_indices]
            application_z_true_data = application_redshift_true_select[application_indices]
            
            application_cluster_id_data = cluster_id[application_cell_id_data]
            application_cluster_count_data = numpy.bincount(application_cluster_id_data, minlength=cluster_size)
            
            application_cluster_z_phot_data = numpy.bincount(application_cluster_id_data, weights=application_z_phot_data, minlength=cluster_size)
            application_cluster_z_phot_data = numpy.divide(application_cluster_z_phot_data, application_cluster_count_data, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=application_cluster_count_data > 0)
            
            # Restriction
            restriction_indices = random_generator.choice(numpy.arange(reference_size), size=reference_size, replace=True)
            
            restriction_z_phot_data = restriction_z_phot_reference[restriction_indices]
            restriction_z_spec_data = restriction_z_spec_reference[restriction_indices]
            restriction_cell_id_data = restriction_cell_id_reference[restriction_indices]
            
            restriction_cluster_id_data = cluster_id[restriction_cell_id_data]
            restriction_cluster_count_data = numpy.bincount(restriction_cluster_id_data, minlength=cluster_size)
            
            restriction_cluster_z_phot_data = numpy.bincount(restriction_cluster_id_data, weights=restriction_z_phot_data, minlength=cluster_size)
            restriction_cluster_z_phot_data = numpy.divide(restriction_cluster_z_phot_data, restriction_cluster_count_data, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=restriction_cluster_count_data > 0)
            
            restriction_cluster_z_spec_data = numpy.bincount(restriction_cluster_id_data, weights=restriction_z_spec_data, minlength=cluster_size)
            restriction_cluster_z_spec_data = numpy.divide(restriction_cluster_z_spec_data, restriction_cluster_count_data, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=restriction_cluster_count_data > 0)
            
            # Filter
            filter_data = (application_cluster_count_data > 0) & (restriction_cluster_count_data > 0)
            
            # Application Mask
            application_cluster_mask = filter_data[application_cluster_id_data]
            application_weight_data = numpy.array(application_cluster_mask, dtype=numpy.float32)
            application_ensemble_indices = numpy.digitize(application_z_true_data, bins=z_grid, right=False) - 1
            
            # Ensemble Cluster
            ensemble_cluster = numpy.zeros((cluster_size, grid_size + 1))
            numpy.add.at(ensemble_cluster, (application_cluster_id_data[application_cluster_mask], application_ensemble_indices[application_cluster_mask]), application_weight_data[application_cluster_mask])
            
            factor = scipy.integrate.trapezoid(x=z_grid, y=ensemble_cluster, axis=1)
            ensemble_cluster = numpy.divide(ensemble_cluster, factor[:, numpy.newaxis], out=numpy.zeros((cluster_size, grid_size + 1)), where=factor[:, numpy.newaxis] > 0)
            
            # Ensemble
            ensemble = numpy.average(ensemble_cluster, axis=0, weights=application_cluster_count_data)
            data_lens[m, n, :] = ensemble / scipy.integrate.trapezoid(x=z_grid, y=ensemble, axis=0)
    
    # Average
    average_lens = numpy.mean(data_lens, axis=1)
    average_lens = average_lens / scipy.integrate.trapezoid(x=z_grid, y=average_lens, axis=1)[:, numpy.newaxis]
    
    # Save
    with h5py.File(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}/TRUTH.hdf5'.format(label, tag, index)), 'w') as file:
        file.create_dataset('data', data=data_lens, dtype=numpy.float32)
        file.create_dataset('average', data=average_lens, dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Summarize Fiducial Truth Lens')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    LABEL = PARSE.parse_args().label
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, LABEL, INDEX, FOLDER)