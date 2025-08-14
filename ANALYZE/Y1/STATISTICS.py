import os
import h5py
import time
import numpy
import scipy
import argparse


def main(tag, rank, label, folder):
    '''
    This function is used to analyze the information of the dataset
    
    Arguments:
        tag (str): The tag of the configuration
        rank (str): The rank of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    print('Rank: {}, Label: {}'.format(rank, label))
    
    # Path
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    os.makedirs(os.path.join(analyze_folder, '{}/STATISTICS/'.format(tag)), exist_ok=True)
    
    # Summarize
    with h5py.File(os.path.join(synthesize_folder, '{}/{}_{}.hdf5'.format(tag, rank, label)), 'r') as file:
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
    
    # Width
    width_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - middle_lens[:, numpy.newaxis]) * average_lens, axis=1))
    width_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - middle_source[:, numpy.newaxis]) * average_source, axis=1))
    
    # Scatter
    scatter_lens = numpy.std(expectation_lens, axis=0) / (1 + middle_lens)
    scatter_source = numpy.std(expectation_source, axis=0) / (1 + middle_source)
    
    # Variation
    variation_lens = numpy.std(deviation_lens, axis=0) / (1 + middle_lens)
    variation_source = numpy.std(deviation_source, axis=0) / (1 + middle_source)
    
    # Correlation Expectation
    correlation_expectation_lens = numpy.corrcoef(expectation_lens, rowvar=False)
    correlation_expectation_source = numpy.corrcoef(expectation_source, rowvar=False)
    
    # Delta
    delta_lens = numpy.random.multivariate_normal(mean=numpy.zeros(bin_lens_size), cov=numpy.cov(expectation_lens, rowvar=False), size=data_size)
    delta_source = numpy.random.multivariate_normal(mean=numpy.zeros(bin_source_size), cov=numpy.cov(expectation_source, rowvar=False), size=data_size)
    
    # Shift
    shift_lens = numpy.zeros((data_size, bin_lens_size, z_size))
    shift_source = numpy.zeros((data_size, bin_source_size, z_size))
    
    for m in range(bin_lens_size):
        z_shift = z_grid[numpy.newaxis, :] - delta_lens[:, m, numpy.newaxis]
        shift_lens[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, average_lens[m, :], extrapolate=True)(z_shift), 0)
    shift_lens = shift_lens / scipy.integrate.trapezoid(x=z_grid, y=shift_lens, axis=2)[:, :, numpy.newaxis]
    
    for m in range(bin_source_size):
        z_shift = z_grid[numpy.newaxis, :] - delta_source[:, m, numpy.newaxis]
        shift_source[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, average_source[m, :], extrapolate=True)(z_shift), 0)
    shift_source = shift_source / scipy.integrate.trapezoid(x=z_grid, y=shift_source, axis=2)[:, :, numpy.newaxis]
    
    # Correlation Deviation
    correlation_deviation_lens = numpy.corrcoef(deviation_lens, rowvar=False)
    correlation_deviation_source = numpy.corrcoef(deviation_source, rowvar=False)
    
    # Zeta
    zeta_lens = numpy.random.multivariate_normal(mean=numpy.zeros(bin_lens_size), cov=numpy.cov(deviation_lens, rowvar=False), size=data_size)
    zeta_source = numpy.random.multivariate_normal(mean=numpy.zeros(bin_source_size), cov=numpy.cov(deviation_source, rowvar=False), size=data_size)
    
    # Scale
    scale_lens = numpy.zeros((data_size, bin_lens_size, z_size))
    scale_source = numpy.zeros((data_size, bin_source_size, z_size))
    
    for m in range(bin_lens_size):
        z_scale = middle_lens[m] + (z_grid[numpy.newaxis, :] - middle_lens[m] - delta_lens[:, m, numpy.newaxis]) / (1 + zeta_lens[:, m, numpy.newaxis] / width_lens[m])
        scale_lens[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, average_lens[m, :], extrapolate=True)(z_scale), 0)
    scale_lens = scale_lens / scipy.integrate.trapezoid(x=z_grid, y=scale_lens, axis=2)[:, :, numpy.newaxis]
    
    for m in range(bin_source_size):
        z_scale = middle_source[m] + (z_grid[numpy.newaxis, :] - middle_source[m] - delta_source[:, m, numpy.newaxis]) / (1 + zeta_source[:, m, numpy.newaxis] / width_source[m])
        scale_source[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, average_source[m, :], extrapolate=True)(z_scale), 0)
    scale_source = scale_source / scipy.integrate.trapezoid(x=z_grid, y=scale_source, axis=2)[:, :, numpy.newaxis]
    
    # Save
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/{}_{}.hdf5'.format(tag, rank, label)), 'w') as file:
        file.create_group('lens')
        file['lens'].create_dataset('zeta', data=zeta_lens)
        file['lens'].create_dataset('delta', data=delta_lens)
        file['lens'].create_dataset('scale', data=scale_lens)
        file['lens'].create_dataset('shift', data=shift_lens)
        file['lens'].create_dataset('width', data=width_lens)
        file['lens'].create_dataset('middle', data=middle_lens)
        file['lens'].create_dataset('scatter', data=scatter_lens)
        file['lens'].create_dataset('deviation', data=deviation_lens)
        file['lens'].create_dataset('variation', data=variation_lens)
        file['lens'].create_dataset('expectation', data=expectation_lens)
        file['lens'].create_dataset('correlation_deviation', data=correlation_deviation_lens)
        file['lens'].create_dataset('correlation_expectation', data=correlation_expectation_lens)
        
        file.create_group('source')
        file['source'].create_dataset('zeta', data=zeta_source)
        file['source'].create_dataset('delta', data=delta_source)
        file['source'].create_dataset('scale', data=scale_source)
        file['source'].create_dataset('shift', data=shift_source)
        file['source'].create_dataset('width', data=width_source)
        file['source'].create_dataset('middle', data=middle_source)
        file['source'].create_dataset('scatter', data=scatter_source)
        file['source'].create_dataset('deviation', data=deviation_source)
        file['source'].create_dataset('variation', data=variation_source)
        file['source'].create_dataset('expectation', data=expectation_source)
        file['source'].create_dataset('correlation_deviation', data=correlation_deviation_source)
        file['source'].create_dataset('correlation_expectation', data=correlation_expectation_source)
    
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
    PARSE.add_argument('--rank', type=str, required=True, help='The rank of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    RANK = PARSE.parse_args().rank
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, RANK, LABEL, FOLDER)