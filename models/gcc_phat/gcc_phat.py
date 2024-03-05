import torch

class gcc_phat():
    
    def __init__(self, output_size=1000):
        self.output_size = output_size

    def __call__(self,x):
        x = torch.fft.fft(x)
        x = x[:,0]*torch.conj(x[:,1])
        #x[500:-500] = 0
        x /= x.abs() + 1e-5
        x = torch.fft.ifft(x).real
        x = torch.concatenate([x[:,x.shape[1]-self.output_size//2:],x[:,:self.output_size//2]],dim=1)
        return x
        
