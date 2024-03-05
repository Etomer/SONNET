import torch

class DatasetTDE(torch.utils.data.Dataset):
    
    def __init__(self, dataset, sampling_freq=16000,speed_of_sound = 343):
        self.sampling_freq = sampling_freq
        self.speed_of_sound = speed_of_sound
        self.X = dataset["input"]
        self.y = dataset["gt"]
        self.pair_idx_map = []
        for i in range(self.X.shape[1]):
            for j in range(i+1,self.X.shape[1]):
                self.pair_idx_map.append((i,j))


    def __getitem__(self,idx):
        pair_idx = idx//self.X.shape[0]
        room_idx = idx % self.X.shape[0]
        mics = self.pair_idx_map[pair_idx]
        
        reX = torch.stack([torch.tensor(self.X[room_idx, mics[0]]),torch.tensor(self.X[room_idx, mics[1]])],axis=0)
        rey = self.y[room_idx,mics[0]] - self.y[room_idx,mics[1]]

        return reX,rey*self.sampling_freq/self.speed_of_sound

    def __len__(self):
        return self.X.shape[0]*(self.X.shape[1]*(self.X.shape[1]-1))//2
