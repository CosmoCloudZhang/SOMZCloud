import os
import umap  # type: ignore[import-not-found]
import h5py
import time
import numpy
import pickle
import argparse


def main(tag, folder):
    '''
    UMAP baseline embedding.
    
    Arguments:
        tag (str) : the tag of the observing conditions
        folder (str) : the base folder containing the datasets
    
    Returns:
        duration (float) : the duration of the process
    '''
    # Start
    index = 0
    start = time.time()
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET')
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/UMAP/'.format(tag)), exist_ok=True)
    
    # Load
    with h5py.File(os.path.join(dataset_folder, '{}/OBSERVATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        observation_dataset = {key: file[key][...] for key in file.keys()}
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/UMAP/INFORM.hdf5'.format(tag)), 'w') as file:
        file.create_group('photometry')
        file['photometry'].create_dataset('redshift', data=observation_dataset['redshift'], dtype=numpy.float32)
        file['photometry'].create_dataset('redshift_true', data=observation_dataset['redshift_true'], dtype=numpy.float32)
        
        file['photometry'].create_dataset('mag_u_lsst', data=observation_dataset['mag_u_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_g_lsst', data=observation_dataset['mag_g_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_r_lsst', data=observation_dataset['mag_r_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_i_lsst', data=observation_dataset['mag_i_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_z_lsst', data=observation_dataset['mag_z_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_y_lsst', data=observation_dataset['mag_y_lsst'], dtype=numpy.float32)
        
        file['photometry'].create_dataset('mag_u_lsst_err', data=observation_dataset['mag_u_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_g_lsst_err', data=observation_dataset['mag_g_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_r_lsst_err', data=observation_dataset['mag_r_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_i_lsst_err', data=observation_dataset['mag_i_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_z_lsst_err', data=observation_dataset['mag_z_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_y_lsst_err', data=observation_dataset['mag_y_lsst_err'], dtype=numpy.float32)
        
        file.create_group('morphology')
        file['morphology'].create_dataset('ra', data=observation_dataset['ra'], dtype=numpy.float32)
        file['morphology'].create_dataset('dec', data=observation_dataset['dec'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('id', data=observation_dataset['id'], dtype=numpy.int32)
        file['morphology'].create_dataset('value', data=observation_dataset['value'], dtype=numpy.int32)
        
        file['morphology'].create_dataset('mu', data=observation_dataset['mu'], dtype=numpy.float32)
        file['morphology'].create_dataset('eta', data=observation_dataset['eta'], dtype=numpy.float32)
        file['morphology'].create_dataset('sigma', data=observation_dataset['sigma'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('major', data=observation_dataset['major'], dtype=numpy.float32)
        file['morphology'].create_dataset('minor', data=observation_dataset['minor'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('major_disk', data=observation_dataset['major_disk'], dtype=numpy.float32)
        file['morphology'].create_dataset('major_bulge', data=observation_dataset['major_bulge'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('radius', data=observation_dataset['radius'], dtype=numpy.float32)
        file['morphology'].create_dataset('ellipticity_disk', data=observation_dataset['ellipticity_disk'], dtype=numpy.float32)
        file['morphology'].create_dataset('ellipticity_bulge', data=observation_dataset['ellipticity_bulge'], dtype=numpy.float32)
        file['morphology'].create_dataset('bulge_to_total_ratio', data=observation_dataset['bulge_to_total_ratio'], dtype=numpy.float32)
    
    # Features
    feature = numpy.stack([
        observation_dataset['mag_i_lsst'], 
        observation_dataset['mag_u_lsst'] - observation_dataset['mag_g_lsst'], 
        observation_dataset['mag_g_lsst'] - observation_dataset['mag_r_lsst'], 
        observation_dataset['mag_r_lsst'] - observation_dataset['mag_i_lsst'], 
        observation_dataset['mag_i_lsst'] - observation_dataset['mag_z_lsst'], 
        observation_dataset['mag_z_lsst'] - observation_dataset['mag_y_lsst'], 
    ], axis=1).astype(numpy.float32)
    
    # Parameters
    number = -1
    distance = 0.1
    neighbors = 30
    components = 2
    metric = 'euclidean'
    

    reducer = umap.UMAP(
        verbose=True,
        n_jobs=number,
        metric=metric, 
        low_memory=True, 
        min_dist=distance, 
        n_neighbors=neighbors,
        n_components=components, 
    )
    
    embedding = reducer.fit_transform(feature).astype(numpy.float32)
    
    model = {
        'metric': metric, 
        'n_jobs': number,
        'reducer': reducer,
        'min_dist': distance, 
        'embedding': embedding,
        'n_neighbors': neighbors,
        'n_components': components
    }
    
    # Save
    with open(os.path.join(dataset_folder, '{}/UMAP/MODEL.pkl'.format(tag)), 'wb') as model_file:
        pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='UMAP baseline embedding')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)