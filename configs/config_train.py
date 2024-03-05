import torch

#dataset
dataset_path = "results/datasets/musan_big.hdf5"
validation_ratio = 0.02

# model
input_size = 10000
block_width = 1024#1024
n_blocks = 10
output_size = 1000
scale_cnn = 4
def get_model(): # implement get_model to return the model you want to train
    #import models.ResNet.model as modellib
    import models.ResNetFFT.model as modellib
    return modellib.ResNetFFT(input_size,block_width, n_blocks, output_size,scale_cnn=scale_cnn)

# training 
n_epochs = 30
batch_size = 4096
device = "cuda:0" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-4
get_optimizer = lambda model : torch.optim.AdamW(model.parameters(), lr = learning_rate)
#get_optimizer = lambda model : torch.optim.SGD(model.parameters(), lr = 5e-3)
loss_fn = torch.nn.CrossEntropyLoss()
batch_between_print = 200
epochs_between_model_save = 2
#save_model_folder = "models/ResNet/checkpoints"
save_model_folder = "models/ResNetFFT/checkpoints"
model_name = "musan_bigger_copy"

# validation
inlier_threshold = 5

# Augmentations
measurement_noise = True
max_measurement_noise_amplitude_ratio = 1 # ratio of max noise amplitude to signal amplitude (uniform distribution between 0 and this value)
