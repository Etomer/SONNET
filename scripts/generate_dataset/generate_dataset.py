import os,sys
sys.path.append(os.getcwd())
import h5py
import glob
import numpy as np
from scipy.io import wavfile
import configs.config_gen as config_gen
from src.simulation import simulate_room
import time

# check audio files input
sound_paths = glob.glob(os.path.join(config_gen.sound_folder,"**/*.wav"),recursive=True)
print(f"Found {len(sound_paths)} '.wav' files in '{config_gen.sound_folder}'")
good_paths = []
for sound_path in sound_paths:
    fs,s = wavfile.read(sound_path)
    if fs == config_gen.fs and len(s) > config_gen.extra_samples_start_for_echo + config_gen.recording_len:
        good_paths.append(sound_path)
sound_paths = good_paths
print(f"Of these {len(sound_paths)} were usuable")

# create dataset
with h5py.File(os.path.join(config_gen.dataset_target_folder ,config_gen.dataset_target_name + ".hdf5"), "w") as hdf5_file:
    
    X = hdf5_file.create_dataset("input",[config_gen.n_rooms_simulate,config_gen.n_mics,config_gen.recording_len] ,dtype="f")
    Y = hdf5_file.create_dataset("gt",[config_gen.n_rooms_simulate,config_gen.n_mics], dtype="f")

    log_logger = -float("inf")
    time_start = time.time()
    for i in range(config_gen.n_rooms_simulate):
        
        sound_path = np.random.choice(sound_paths, 1)[0]
        fs,signal = wavfile.read(sound_path)
        sounds, toas = simulate_room(signal)

        X[i] = sounds
        Y[i] = toas

        if np.floor(np.log((i+1)/config_gen.n_rooms_simulate)) > log_logger:
            print(f'{i+1} / {config_gen.n_rooms_simulate} after: {time.time() - time_start:.1f} seconds')
            log_logger = np.floor(np.log((i+1)/config_gen.n_rooms_simulate))



