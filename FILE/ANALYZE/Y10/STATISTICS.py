import os
import h5py
import time
import numpy
import scipy
import argparse


def main(tag, type, label, folder):
    '''
    This function is used to analyze the information of the dataset
    
    Arguments:
        tag (str): The tag of the configuration
        type (str): The type of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    print('Type: {}, Label: {}'.format(type, label))
    
    # Path
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    os.makedirs(os.path.join(analyze_folder, '{}/STATISTICS/'.format(tag)), exist_ok=True)
    
    # Summarize
    with h5py.File(os.path.join(synthesize_folder, '{}/{}_{}.hdf5'.format(tag, type, label)), 'r') as file:
        data_lens = file['lens']['data'][...]
        data_source = file['source']['data'][...]
        
        average_lens = file['lens']['average'][...]
        average_source = file['source']['average'][...]
    
    data_size, bin_lens_size, z_size = data_lens.shape
    data_size, bin_source_size, z_size = data_source.shape
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, z_size)
    
    # Expectation
    expectation_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * data_lens, axis=2)
    expectation_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * data_source, axis=2)
    
    # Middle
    middle_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * average_lens, axis=1)
    middle_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * average_source, axis=1)
    
    # Deviation
    deviation_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - expectation_lens[:, :, numpy.newaxis]) * data_lens, axis=2))
    deviation_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - expectation_source[:, :, numpy.newaxis]) * data_source, axis=2))
    
    # Scatter
    scatter_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - middle_lens[:, numpy.newaxis]) * average_lens, axis=1))
    scatter_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - middle_source[:, numpy.newaxis]) * average_source, axis=1))
    
    # Scale
    scale_lens = numpy.std(expectation_lens, axis=0) / (1 + middle_lens)
    scale_source = numpy.std(expectation_source, axis=0) / (1 + middle_source)
    
    # Variation
    variation_lens = numpy.std(deviation_lens, axis=0) / (1 + middle_lens)
    variation_source = numpy.std(deviation_source, axis=0) / (1 + middle_source)
    
    # Correlation
    correlation_lens = numpy.corrcoef(expectation_lens, rowvar=False)
    correlation_source = numpy.corrcoef(expectation_source, rowvar=False)
    
    # Delta
    delta_lens = numpy.random.multivariate_normal(mean=numpy.zeros(bin_lens_size), cov=numpy.cov(expectation_lens, rowvar=False), size=data_size)
    delta_source = numpy.random.multivariate_normal(mean=numpy.zeros(bin_source_size), cov=numpy.cov(expectation_source, rowvar=False), size=data_size)
    
    # Shift
    shift_lens = numpy.zeros((data_size, bin_lens_size, z_size))
    shift_source = numpy.zeros((data_size, bin_source_size, z_size))
    
    for m in range(bin_lens_size):
        shift_lens[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, average_lens[m, :], extrapolate=True)(z_grid[numpy.newaxis, :] - delta_lens[:, m, numpy.newaxis]), 0)
    shift_lens = shift_lens / scipy.integrate.trapezoid(x=z_grid, y=shift_lens, axis=2)[:, :, numpy.newaxis]
    
    for m in range(bin_source_size):
        shift_source[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, average_source[m, :], extrapolate=True)(z_grid[numpy.newaxis, :] - delta_source[:, m, numpy.newaxis]), 0)
    shift_source = shift_source / scipy.integrate.trapezoid(x=z_grid, y=shift_source, axis=2)[:, :, numpy.newaxis]
    
    # Save
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/{}_{}.hdf5'.format(tag, type,label)), 'w') as file:
        file.create_group('lens')
        file['lens'].create_dataset('delta', data=delta_lens)
        file['lens'].create_dataset('scale', data=scale_lens)
        file['lens'].create_dataset('shift', data=shift_lens)
        file['lens'].create_dataset('middle', data=middle_lens)
        file['lens'].create_dataset('scatter', data=scatter_lens)
        file['lens'].create_dataset('deviation', data=deviation_lens)
        file['lens'].create_dataset('variation', data=variation_lens)
        file['lens'].create_dataset('expectation', data=expectation_lens)
        file['lens'].create_dataset('correlation', data=correlation_lens)
        
        file.create_group('source')
        file['source'].create_dataset('delta', data=delta_source)
        file['source'].create_dataset('scale', data=scale_source)
        file['source'].create_dataset('shift', data=shift_source)
        file['source'].create_dataset('middle', data=middle_source)
        file['source'].create_dataset('scatter', data=scatter_source)
        file['source'].create_dataset('deviation', data=deviation_source)
        file['source'].create_dataset('variation', data=variation_source)
        file['source'].create_dataset('expectation', data=expectation_source)
        file['source'].create_dataset('correlation', data=correlation_source)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Analysis Statistics')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--type', type=str, required=True, help='The type of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    TYPE = PARSE.parse_args().type
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, TYPE, LABEL, FOLDER)