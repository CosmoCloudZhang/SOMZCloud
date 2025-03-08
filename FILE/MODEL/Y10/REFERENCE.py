import os
import time
import h5py
import numpy
import scipy
import argparse

import scipy.interpolate


def main(tag, index, folder):
    '''
    Define the lens and source reference
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        float: The duration of the process
    '''
    # Start
    start = time.time()
    print('Index:{}'.format(index))
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(model_folder, '{}/REFERENCE/'.format(tag)), exist_ok=True)
    
    os.makedirs(os.path.join(model_folder, '{}/LENS/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(model_folder, '{}/LENS/LENS{}/'.format(tag, index)), exist_ok=True)
    
    os.makedirs(os.path.join(model_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/'.format(tag, index)), exist_ok=True)
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    
    z1_source = 0.05
    z2_source = 2.95
    
    z1 = 0.0
    z2 = 3.0
    
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    mesh_size = 3000
    z_delta = (z2 - z1) / mesh_size
    z_mesh = numpy.linspace(z1, z2, mesh_size + 1)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_magnitude = file['photometry']['mag_i_lsst'][...]
        combination_redshift_true = file['photometry']['redshift_true'][...]
    combination_size = len(combination_magnitude)
    
    # Bin
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_lens = file['bin_lens'][...]
        bin_source = file['bin_source'][...]
    
    lens_size = len(bin_lens) - 1
    source_size = len(bin_source) - 1
    
    # Evaluate
    chunk_size = 100000
    z_phot = numpy.zeros(combination_size, dtype=numpy.float32)
    evaluator = h5py.File(os.path.join(model_folder, '{}/EVALUATE/EVALUATE{}.hdf5'.format(tag, index)), 'r')
    
    for m in range(combination_size // chunk_size + 1):
        begin = m * chunk_size
        stop = min((m + 1) * chunk_size, combination_size)
        
        if begin < stop:
            z_pdf = scipy.interpolate.CubicSpline(x=z_grid, y=evaluator['data']['yvals'][begin: stop].astype(numpy.float32), axis=1, bc_type='natural', extrapolate=False)(z_mesh)
            z_pdf = z_pdf / numpy.sum(z_pdf, axis=1, keepdims=True) / z_delta
            z_phot[begin: stop] = z_mesh[numpy.argmax(z_pdf, axis=1)]
    
    # Reference
    slope = 4.0
    intercept = 18.5
    reference_source = (z1_source <= z_phot) & (z_phot < z2_source)
    reference_lens = (z1_lens <= z_phot) & (z_phot < z2_lens) & (combination_magnitude < slope * z_phot + intercept)
    
    # Metric
    z_phot_lens = z_phot[reference_lens]
    z_true_lens = combination_redshift_true[reference_lens]
    
    sigma_lens = numpy.abs(z_phot_lens - z_true_lens) / (1 + z_true_lens)
    nmad_lens = 1.4826 * numpy.median(numpy.abs(sigma_lens - numpy.median(sigma_lens)))
    
    fraction_lens = len(sigma_lens[sigma_lens > 0.15]) / len(sigma_lens) * 100
    percentile_lens = len(sigma_lens[numpy.abs(z_phot_lens - z_true_lens) > 1.0]) / len(sigma_lens) * 100
    
    z_phot_source = z_phot[reference_source]
    z_true_source = combination_redshift_true[reference_source]
    
    sigma_source = numpy.abs(z_phot_source - z_true_source) / (1 + z_true_source)
    nmad_source = 1.4826 * numpy.median(numpy.abs(sigma_source - numpy.median(sigma_source)))
    
    fraction_source = len(sigma_source[sigma_source > 0.15]) / len(sigma_source) * 100
    percentile_source = len(sigma_source[numpy.abs(z_phot_source - z_true_source) > 1.0]) / len(sigma_source) * 100
    
    # Save
    with h5py.File(os.path.join(model_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('z_phot', data=z_phot, dtype=numpy.float32)
        
        file.create_dataset('bin_lens', data=bin_lens, dtype=numpy.float32)
        file.create_dataset('reference_lens', data=reference_lens, dtype=bool)
        
        file.create_dataset('nmad_lens', data=nmad_lens, dtype=numpy.float32)
        file.create_dataset('fraction_lens', data=fraction_lens, dtype=numpy.float32)
        file.create_dataset('percentile_lens', data=percentile_lens, dtype=numpy.float32)
        
        file.create_dataset('bin_source', data=bin_source, dtype=numpy.float32)
        file.create_dataset('reference_source', data=reference_source, dtype=bool)
        
        file.create_dataset('nmad_source', data=nmad_source, dtype=numpy.float32)
        file.create_dataset('fraction_source', data=fraction_source, dtype=numpy.float32)
        file.create_dataset('percentile_source', data=percentile_source, dtype=numpy.float32)
    
    # Lens bin
    nmad_lens_bin = numpy.zeros(lens_size, dtype=numpy.float32)
    fraction_lens_bin = numpy.zeros(lens_size, dtype=numpy.float32)
    percentile_lens_bin = numpy.zeros(lens_size, dtype=numpy.float32)
    reference_lens_bin = numpy.ones((lens_size, combination_size), dtype=bool)
    
    for m in range(len(bin_lens) - 1):
        reference_lens_bin[m, :] = reference_lens & (bin_lens[m] <= z_phot) & (z_phot < bin_lens[m + 1])
        
        if numpy.sum(reference_lens_bin[m, :]) > 0:
            z_phot_lens_bin = z_phot[reference_lens_bin[m, :]]
            z_true_lens_bin = combination_redshift_true[reference_lens_bin[m, :]]
            
            sigma_lens_bin = numpy.abs(z_phot_lens_bin - z_true_lens_bin) / (1 + z_true_lens_bin)
            nmad_lens_bin[m] = 1.4826 * numpy.median(numpy.abs(sigma_lens_bin - numpy.median(sigma_lens_bin)))
            
            fraction_lens_bin[m] = len(sigma_lens_bin[sigma_lens_bin > 0.15]) / len(sigma_lens_bin) * 100
            percentile_lens_bin[m] = len(sigma_lens_bin[numpy.abs(z_phot_lens_bin - z_true_lens_bin) > 1.0]) / len(sigma_lens_bin) * 100
        else:
            nmad_lens_bin[m] = numpy.nan
            fraction_lens_bin[m] = numpy.nan
            percentile_lens_bin[m] = numpy.nan
    
    with h5py.File(os.path.join(model_folder, '{}/LENS/LENS{}/REFERENCE.hdf5'.format(tag, index)), 'w') as file:    
        file.create_dataset('reference', data=reference_lens_bin, dtype=bool)
        
        file.create_dataset('nmad', data=nmad_lens_bin, dtype=numpy.float32)
        file.create_dataset('fraction', data=fraction_lens_bin, dtype=numpy.float32)
        file.create_dataset('percentile', data=percentile_lens_bin, dtype=numpy.float32)
    
    # Source
    nmad_source_bin = numpy.zeros(source_size, dtype=numpy.float32)
    fraction_source_bin = numpy.zeros(source_size, dtype=numpy.float32)
    percentile_source_bin = numpy.zeros(source_size, dtype=numpy.float32)
    reference_source_bin = numpy.ones((source_size, combination_size), dtype=bool)
    
    for m in range(len(bin_source) - 1):
        reference_source_bin[m, :] = reference_source & (bin_source[m] <= z_phot) & (z_phot < bin_source[m + 1])
        
        if numpy.sum(reference_source_bin[m, :]) > 0:
            z_phot_source_bin = z_phot[reference_source_bin[m, :]]
            z_true_source_bin = combination_redshift_true[reference_source_bin[m, :]]
            
            sigma_source_bin = numpy.abs(z_phot_source_bin - z_true_source_bin) / (1 + z_true_source_bin)
            nmad_source_bin[m] = 1.4826 * numpy.median(numpy.abs(sigma_source_bin - numpy.median(sigma_source_bin)))
            
            fraction_source_bin[m] = len(sigma_source_bin[sigma_source_bin > 0.15]) / len(sigma_source_bin) * 100
            percentile_source_bin[m] = len(sigma_source_bin[numpy.abs(z_phot_source_bin - z_true_source_bin) > 1.0]) / len(sigma_source_bin) * 100
        else:
            nmad_source_bin[m] = numpy.nan
            fraction_source_bin[m] = numpy.nan
            percentile_source_bin[m] = numpy.nan
    
    with h5py.File(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/REFERENCE.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('reference', data=reference_source_bin, dtype=bool)
        
        file.create_dataset('nmad', data=nmad_source_bin, dtype=numpy.float32)
        file.create_dataset('fraction', data=fraction_source_bin, dtype=numpy.float32)
        file.create_dataset('percentile', data=percentile_source_bin, dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Model Reference')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)