import torch

#dataset
dataset_path = "results/datasets/musan.hdf5"
validation_ratio = 0.02

# model
input_size = 10000
block_width = 256#512#1024#1500#1024#1024
n_blocks = 2
output_size = 1000
scale_cnn = 1#6#4
def get_model(): # implement get_model to return the model you want to train
    #import models.ResNet.model as modellib
    import models.ResNetFFT.model as modellib
    return modellib.ResNetFFT(input_size,block_width, n_blocks, output_size,scale_cnn=scale_cnn)

# training 
n_epochs = 20
batch_size = 4096
device = "cuda:0" if torch.cuda.is_available() else "cpu"
learning_rate = 3e-4
get_optimizer = lambda model : torch.optim.AdamW(model.parameters(), lr = learning_rate)

loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
batch_between_print = 200
epochs_between_model_save = 5

save_model_folder = "models/ResNetFFT/checkpoints"
model_name = "sonnet"

# validation
inlier_threshold = 5

# Augmentations
measurement_noise = True
max_measurement_noise_amplitude_ratio = 0.1 # ratio of max noise amplitude to signal amplitude (uniform distribution between 0 and this value)