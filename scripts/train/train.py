import os,sys
sys.path.append(os.getcwd())
import torch, h5py
from src.DatasetTDE import DatasetTDE
import configs.config_train as config_train
from src.DatasetTdoa20201016 import DatasetTdoa20201016

# Fix random seed for reproducability
torch.manual_seed(37)

# Prepare dataset
dataset =  DatasetTDE(h5py.File(config_train.dataset_path, "r"))
#dataset = DatasetTdoa20201016()
dataset_train, dataset_val = torch.utils.data.random_split(dataset, [1 - config_train.validation_ratio, config_train.validation_ratio])

#dataset_train, dataset_val,_ = torch.utils.data.random_split(dataset, [0.05, 0.01, 0.94])
dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=config_train.batch_size, shuffle=True,num_workers=4)
dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=config_train.batch_size, shuffle=True,num_workers=4)

# prepare model
model = config_train.get_model()
model = model.to(config_train.device)
optimizer = config_train.get_optimizer(model)

def train(dataloader, model, loss_fn, optimizer):
    model.train()

    losses = []
    counts = []
    for batch, (X, y) in enumerate(dataloader):
   
        counts.append(X.shape[0])
        y = (y + config_train.output_size//2).long()
        y[y >= config_train.output_size] = config_train.output_size-1
        y[y < 0] = 0
        X = X.float()
        X, y = X.to(config_train.device), y.to(config_train.device)
        
        if config_train.measurement_noise:
            X = X + torch.randn(X.shape,device=config_train.device)*X.std(dim=2,keepdim=True)*torch.rand(X.shape[0],device=config_train.device).unsqueeze(1).unsqueeze(2)*config_train.max_measurement_noise_amplitude_ratio

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.cpu().detach().item())

        if batch % config_train.batch_between_print == config_train.batch_between_print-1:
            loss = sum([a*b for a,b in zip(losses,counts)])/sum(counts)
            losses = []
            counts = []
            #loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{batch+1:>5d}/{len(dataloader):>5d}]")

def val(dataloader, model, loss_fn):
    model.eval()
    with torch.no_grad():
        counts = []
        reasonables = []
        losses = []
        
        for batch, (X, y) in enumerate(dataloader):
            counts.append(X.shape[0])
            
            y = (y + config_train.output_size//2).long()
            y[y >= config_train.output_size] = config_train.output_size-1
            y[y < 0] = 0
            X = X.float()
            X, y = X.to(config_train.device), y.to(config_train.device)

            #if config_train.measurement_noise:
            #    X = X + torch.randn(X.shape,device=config_train.device)*X.std(dim=2,keepdim=True)*torch.rand(X.shape[0],device=config_train.device).unsqueeze(1).unsqueeze(2)*config_train.max_measurement_noise_amplitude_ratio

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            losses.append(loss.cpu().item())
            
            # Compute accuracy bellow some threshold too
            temp = (pred.argmax(dim=1) - y).abs() < config_train.inlier_threshold
            reasonables.append(temp.sum()/temp.numel())
        
        fin_loss = 0
        fin_reasonable = 0
        for i,l in enumerate(losses):
            fin_loss += l*counts[i]
            fin_reasonable += reasonables[i]*counts[i]
        fin_loss /= sum(counts)
        fin_reasonable /= sum(counts) 

        print(fin_loss)
        print(f'Ratio within 1 dm : {fin_reasonable:.2f}')


for epoch in range(config_train.n_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(dl_train,model, config_train.loss_fn, optimizer)
    print("val sim:")
    val(dl_val,model,config_train.loss_fn)
    
    if epoch % config_train.epochs_between_model_save == 0:
        torch.save(model,os.path.join(config_train.save_model_folder,config_train.model_name + "_" + str(epoch)+ ".pth"))