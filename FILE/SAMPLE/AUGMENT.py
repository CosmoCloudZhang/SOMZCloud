import os
import time
import h5py
import numpy
import argparse

def main(path):
    
    # Data
    data_path = os.path.join(path, 'DATA/')
    
    augment_file = {}
    testing_file = h5py.File(os.path.join(data_path, 'SAMPLE/TESTING.hdf5'), 'r')
    training_file = h5py.File(os.path.join(data_path, 'SAMPLE/TRAINING.hdf5'), 'r')
    validation_file = h5py.File(os.path.join(data_path, 'SAMPLE/VALIDATION.hdf5'), 'r')
    
    try:
        # Redshift
        redshift_testing = testing_file['redshift_true'][:].astype('float32')
        redshift_training = training_file['redshift_true'][:].astype('float32')
        redshift_validation = validation_file['redshift_true'][:].astype('float32')
        augment_file['redshift'] = numpy.concatenate([redshift_testing, redshift_training, redshift_validation])
        
        # U
        u_testing = testing_file['u_mag'][:].astype('float32')
        u_training = training_file['u_mag'][:].astype('float32')
        u_validation = validation_file['u_mag'][:].astype('float32')
        augment_file['mag_u_lsst'] = numpy.concatenate([u_testing, u_training, u_validation])
        
        u_err_testing = testing_file['u_mag_err'][:].astype('float32')
        u_err_training = training_file['u_mag_err'][:].astype('float32')
        u_err_validation = validation_file['u_mag_err'][:].astype('float32')
        augment_file['mag_err_u_lsst'] = numpy.concatenate([u_err_testing, u_err_training, u_err_validation])
        
        # G
        g_testing = testing_file['g_mag'][:].astype('float32')
        g_training = training_file['g_mag'][:].astype('float32')
        g_validation = validation_file['g_mag'][:].astype('float32')
        augment_file['mag_g_lsst'] = numpy.concatenate([g_testing, g_training, g_validation])
        
        g_err_testing = testing_file['g_mag_err'][:].astype('float32')
        g_err_training = training_file['g_mag_err'][:].astype('float32')
        g_err_validation = validation_file['g_mag_err'][:].astype('float32')
        augment_file['mag_err_g_lsst'] = numpy.concatenate([g_err_testing, g_err_training, g_err_validation])
        
        # R
        r_testing = testing_file['r_mag'][:].astype('float32')
        r_training = training_file['r_mag'][:].astype('float32')
        r_validation = validation_file['r_mag'][:].astype('float32')
        augment_file['mag_r_lsst'] = numpy.concatenate([r_testing, r_training, r_validation])
        
        r_err_testing = testing_file['r_mag_err'][:].astype('float32')
        r_err_training = training_file['r_mag_err'][:].astype('float32')
        r_err_validation = validation_file['r_mag_err'][:].astype('float32')
        augment_file['mag_err_r_lsst'] = numpy.concatenate([r_err_testing, r_err_training, r_err_validation])
        
        # I
        i_testing = testing_file['i_mag'][:].astype('float32')
        i_training = training_file['i_mag'][:].astype('float32')
        i_validation = validation_file['i_mag'][:].astype('float32')
        augment_file['mag_i_lsst'] = numpy.concatenate([i_testing, i_training, i_validation])
        
        i_err_testing = testing_file['i_mag_err'][:].astype('float32')
        i_err_training = training_file['i_mag_err'][:].astype('float32')
        i_err_validation = validation_file['i_mag_err'][:].astype('float32')
        augment_file['mag_err_i_lsst'] = numpy.concatenate([i_err_testing, i_err_training, i_err_validation])
        
        # Z
        z_testing = testing_file['z_mag'][:].astype('float32')
        z_training = training_file['z_mag'][:].astype('float32')
        z_validation = validation_file['z_mag'][:].astype('float32')
        augment_file['mag_z_lsst'] = numpy.concatenate([z_testing, z_training, z_validation])
        
        z_err_testing = testing_file['z_mag_err'][:].astype('float32')
        z_err_training = training_file['z_mag_err'][:].astype('float32')
        z_err_validation = validation_file['z_mag_err'][:].astype('float32')
        augment_file['mag_err_z_lsst'] = numpy.concatenate([z_err_testing, z_err_training, z_err_validation])
        
        # Y
        y_testing = testing_file['y_mag'][:].astype('float32')
        y_training = training_file['y_mag'][:].astype('float32')
        y_validation = validation_file['y_mag'][:].astype('float32')
        augment_file['mag_y_lsst'] = numpy.concatenate([y_testing, y_training, y_validation])
        
        y_err_testing = testing_file['y_mag_err'][:].astype('float32')
        y_err_training = training_file['y_mag_err'][:].astype('float32')
        y_err_validation = validation_file['y_mag_err'][:].astype('float32')
        augment_file['mag_err_y_lsst'] = numpy.concatenate([y_err_testing, y_err_training, y_err_validation])
    
    finally:
        testing_file.close()
        training_file.close()
        validation_file.close()
    
    # Select
    for key, value in augment_file.items():
        value[~numpy.isfinite(value)] = 99.0
        augment_file[key] = value
    
    # Save
    with h5py.File(os.path.join(data_path, 'SAMPLE/AUGMENT_SAMPLE.hdf5'), 'w') as file:
        group = file.create_group('photometry')
        for key, value in augment_file.items():
            group.create_dataset(key, data=value)
    
if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Augmentation sample.')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder.')
    
    PATH = PARSE.parse_args().path
    START = time.time()
    
    main(PATH)
    END = time.time()
    print('Time: {:.2f} minutes'.format((END - START) / 60))