import os
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Models to evaluate
from models.gcc_phat.gcc_phat import gcc_phat

def get_models():
    models = {
        
        "learned_model_small" : torch.load("models/ResNetFFT/checkpoints/musan_bigger_model_small_17.pth").to(device),
        "learned_model" : torch.load("models/ResNetFFT/checkpoints/musan_bigger_45.pth").to(device),
        "learned_model_large" : torch.load("models/ResNetFFT/checkpoints/musan_bigger_14.pth").to(device),
 
        "gcc_phat" : gcc_phat()
    }
    return models

# Simulated evaluation
reverberations_levels = [0.05 + i/10.0 for i in range(9)]
snr_at_reverb_measurement = 10
n_rooms_simulate = 100
n_mics = 50
reflection_coeff_min = 0.05
reflection_coeff_max = 0.99
scatter_coeff = 0.15
room_max_size = 10
room_min_size = 1
extra_samples_start_for_echo = 2000 #samples


snr_levels = [-30 + 5*t for t in range(13)]
reverb_at_snr_dependency_measuring = [i for i,r in enumerate(reverberations_levels) if r == 0.55][0]

directivity = False

sound_source_max_speed = 5 # m/s
sound_source_locations_per_recording = 30
fs = 16000

sound_folder = os.path.join("data","musan")
dataset_target_folder = os.path.join("results","datasets")
dataset_target_name = "eval"



