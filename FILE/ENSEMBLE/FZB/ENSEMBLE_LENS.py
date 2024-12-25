import os
import time
import h5py
import numpy
import scipy
import argparse


def main(folder):
    '''
    Compute the ensemble tomographic redshift distributions for the lens galaxies.
    
    Arguments:
        folder (str): The base folder containing the datasets.
    
    Returns:
        duration (float): The duration of the computation in minutes.
    '''
    # Start
    start = time.time()
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    ensemble_folder = os.path.join(folder, 'ENSEMBLE/')
    
    os.makedirs(ensemble_folder, exist_ok=True)
    os.makedirs(os.path.join(ensemble_folder, 'LENS/'), exist_ok=True)
    
    # Load
    data = []  
    for i in range(len([name for name in os.listdir(os.path.join(fzb_folder, 'LENS/')) if 'LENS' in name])):
    
        data.append([])
        for j in range(len([name for name in os.listdir(os.path.join(fzb_folder, 'LENS/LENS{}/'.format(i + 1))) if 'SUMMARIZE' in name])):
            
            with h5py.File(os.path.join(fzb_folder, 'LENS/LENS{}/SUMMARIZE{}.hdf5'.format(i + 1, j + 1)), 'r') as file:
                data[i].append(file['data']['pdfs'][:].astype(numpy.float32))
    data = numpy.array(data)
    
    length, width, height, size = numpy.shape(data)
    number = length * height
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, size)
    
    # Ensemble
    ensemble = numpy.zeros((number, width, size))
    for k in range(number):
        
        n = numpy.arange(length, dtype=numpy.int32)
        m = numpy.random.choice(numpy.arange(height, dtype=numpy.int32), size=length, replace=True)
        
        alpha = numpy.random.dirichlet(numpy.ones(length), size=1).flatten()
        beta = numpy.random.dirichlet(alpha, size=1).flatten()
        
        value = numpy.maximum(numpy.sum(beta[:, numpy.newaxis, numpy.newaxis] * data[n, :, m, :], axis=0), 0.0)
        value = value / scipy.integrate.trapezoid(y=value, x=z_grid, axis=1)[:, numpy.newaxis]
        ensemble[k, :, :] = value
    
    with h5py.File(os.path.join(ensemble_folder, 'LENS/FZB_ENSEMBLE.hdf5'), 'w') as file:
        file.create_dataset('sample', data=ensemble, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Ensemble')
    PARSE.add_argument('--folder', dest='folder', type=str, help='The base folder containing the datasets.')
    
    # Parse
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(FOLDER)
