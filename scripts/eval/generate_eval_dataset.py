import os, sys
sys.path.append(os.getcwd())
import glob
from scipy.io import wavfile
import configs.config_gen as config_gen
import numpy as np
from src.simulation import simulate_room_series_with_different_reverb
import configs.config_eval as config_eval
import h5py

np.random.seed(37)

sound_paths = glob.glob(os.path.join(config_eval.sound_folder,"**/*.wav"),recursive=True)
print(f"Found {len(sound_paths)} '.wav' files in '{config_eval.sound_folder}'")
good_paths = []
for sound_path in sound_paths:
    fs,s = wavfile.read(sound_path)
    if fs == config_gen.fs and len(s) > config_gen.extra_samples_start_for_echo + config_gen.recording_len:
        good_paths.append(sound_path)
sound_paths = good_paths
print(f"Of these {len(sound_paths)} were usuable")

dataset_path = os.path.join(config_eval.dataset_target_folder ,config_eval.dataset_target_name + ".hdf5")
if os.path.isfile(dataset_path):
    print("Evaluation '" + dataset_path + "' dataset already exists. So it will not be generated again")
else:

    with h5py.File(dataset_path, "w") as hdf5_file:
        
        X = hdf5_file.create_dataset("input",[config_eval.n_rooms_simulate,len(config_eval.reverberations_levels),config_eval.n_mics,config_eval.recording_len] ,dtype="f")
        Y = hdf5_file.create_dataset("gt",[config_eval.n_rooms_simulate,config_eval.n_mics], dtype="f")

        for i in range(config_eval.n_rooms_simulate):
            print(f'{i} / {config_eval.n_rooms_simulate}')
            sound_path = np.random.choice(sound_paths, 1)[0]
            fs,signal = wavfile.read(sound_path)
            sim_res = simulate_room_series_with_different_reverb(signal, config_eval.reverberations_levels)
            
            for j, (sounds, toas) in enumerate(sim_res):
                X[i,j] = sounds
            Y[i] = toas