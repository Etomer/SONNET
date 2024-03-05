import os

recording_len = 10000 #samples
extra_samples_start_for_echo = 2000 #samples

n_rooms_simulate = int(4e4)

n_mics = 50
reflection_coeff_min = 0.05
reflection_coeff_max = 0.99
scatter_coeff = 0.15
room_max_size = 10
room_min_size = 1

# Folder containing sound to use for simulation. Will use all .wav files contained in folder and subfolders.
#sound_folder = os.path.join("data","musan","music")
sound_folder = os.path.join("data","musan")

fs = 16000 # requires all .wav files to have this sampling frequency

directivity = False

sound_source_max_speed = 5 # m/s
sound_source_locations_per_recording = 30 # number of locations to simulate the sound source at (instead of moving the source) Note: this is inculding the extra_samples_start_for_echo

# Storing data set after:
dataset_target_folder = os.path.join("results","datasets")
dataset_target_name = "musan_bigger"