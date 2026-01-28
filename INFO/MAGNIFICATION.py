import os
import json
import time
import argparse


def main(folder):
    '''
    Store the fiducial values of magnification bias
    
    Arguments:
        folder (str): The base folder of the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    info_folder = os.path.join(folder, 'INFO/')
    
    # Magnification
    magnification = {
        'Y1': [0.106, 0.244, 0.307, 0.223, 0.251],
        'Y10': [0.049, 0.143, 0.260, 0.230, 0.284, 0.324, 0.208, 0.242, 0.246, 0.255],
    }
    
    # Save
    with open(os.path.join(info_folder, 'MAGNIFICATION.json'), 'w') as file:
        json.dump(magnification, file, indent=4)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info Magnification')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(FOLDER)