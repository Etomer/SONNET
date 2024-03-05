import os
import numpy as np
import torch
import scipy.io.wavfile as wavfile
import pandas as pd

class DatasetTdoa20201016(torch.utils.data.Dataset):

    # Take ~10s to initialize dataset
    def __init__(self,
            tdoa_20201016_path = os.path.join("data","tdoa_20201016"),
            recording_len = 10000,
            target_sampling_rate = 16000,
            overlap_ratio = 5/6, # 5/6 overlap between different windows
            cutoff_quiet = 1e7, 
            ):
        super().__init__()
        #tdoa_20201016 specific constants
        self.recorded_sampling_rate = recording_len
        self.target_sampling_rate = target_sampling_rate
        self.overlap_ratio = overlap_ratio
        self.cutoff_quiet = cutoff_quiet
        self.n_mics = 12
        self.recorded_sampling_rate = 96000
        usable_experiments = [
        'music_0010',
        'music_0012',
        'music_0011',
        'music_0008',
        'chirp_0004',
        'chirp_0002',
        'music_0014',
        'metronom_0022',
        'metronom_0021',
        'chirp_0001',
        'iregchirp_0006',
        'iregchirp_0007',
        'music_0009',
        'music_0015',
        ]
        usuable_gt_time = {
            'music_0010' : [7,63],
            'music_0012' : [7,69],
            'music_0011' : [7,65],
            'music_0008' : [7,49],
            'chirp_0004' : [4,45],
            'chirp_0002' : [4,30],
            'music_0014' : [12,82],
            'metronom_0022' : [28,41],
            'metronom_0021' : [6,26],
            'chirp_0001' : [0,47],
            'iregchirp_0006' : [5,55],
            'iregchirp_0007' : [6,60],
            'music_0009' : [8,65],
            'music_0015' : [8,79],
        }
        self.speed_of_sound = 343

        self.sounds = {}
        self.times = {}
        self.gt_toa = {}
        self.exp_lens = []
        self.exp_names = []

        for exp in usable_experiments:
            #exp = self.usable_experiments[10]
            exp_path = os.path.join(tdoa_20201016_path, "data", exp)

            # load sound
            exp_sounds = []
            for i in range(self.n_mics):
                fs, temp = wavfile.read(os.path.join(exp_path,"Track " + str(i+1) + ".wav"))
                exp_sounds.append(temp)
            exp_sounds = np.stack(exp_sounds)

            #load ground truth
            df = pd.read_csv(os.path.join(exp_path, "gt_positions.csv")) # Note time-column = 0 when audio-recordings started

            dims = ["x","y","z"]
            time = df["time"].to_numpy() # time = 0 correspond to start of sound-recording
            senders = df[["speaker" + "_" +dim for dim in dims]].to_numpy()
            receivers = np.zeros((self.n_mics,3))
            for i in range(self.n_mics):
                for j,dim in enumerate(["x","y","z"]):
                    temp = df["mic" + str(i+1) + "_" + dim]
                    receivers[i,j] = temp[temp.notnull()].median()


            if self.recorded_sampling_rate/target_sampling_rate - self.recorded_sampling_rate//target_sampling_rate == 0:
                downsampling_rate = self.recorded_sampling_rate//target_sampling_rate
                sample_shifts_between_windows = round((1 - overlap_ratio)*recording_len*downsampling_rate)
                sample_shifts_between_windows = 1 + sample_shifts_between_windows//downsampling_rate*downsampling_rate # make sample_shift_between_window be relative prime to downsampling_rate
                starts = np.arange(0,exp_sounds.shape[1]-recording_len*downsampling_rate,sample_shifts_between_windows)
                sound_windows = np.stack([exp_sounds[:,start:start+recording_len*downsampling_rate:downsampling_rate] for start in starts])
                time_windows = (starts + recording_len*downsampling_rate/2)/self.recorded_sampling_rate
                sender_windows = np.stack([np.interp(time_windows, time, senders[:,i]) for i in range(3)]).T
                gt_toa_windows = np.stack([np.linalg.norm(sender_windows - receivers[i,:],axis=1) for i in range(self.n_mics)]).T
                good_gt_idx = np.all(np.isfinite(gt_toa_windows),axis=1) # remove idx with nan as gt
                good_time_idx = (time_windows > usuable_gt_time[exp][0]) & (time_windows < usuable_gt_time[exp][1] ) #...and also remove windows outside of usuable_gt times
                has_sound_idx = sound_windows.std(axis=(1,2)) > cutoff_quiet
                # ...and check that sound is playing
                good_idx = good_gt_idx & good_time_idx & has_sound_idx
                self.sounds[exp] = np.float32(sound_windows[good_idx])
                self.times[exp] = time_windows[good_idx]
                self.gt_toa[exp] = gt_toa_windows[good_idx]
                self.exp_lens.append(time_windows[good_idx].shape[0])
                self.exp_names.append(exp)
            else:
                raise Exception("Not implemented for non-integer downsampling yet")
        
        self.pair_idx_to_component_idx = []
        for i in range(self.n_mics):
            for j in range(i+1,self.n_mics):
                self.pair_idx_to_component_idx.append((i,j))
        self.n_pairs = len(self.pair_idx_to_component_idx) #(self.n_mics*(self.n_mics + 1))//2
        self.exp_lens = np.array(self.exp_lens)*self.n_pairs
        self.c_exp_lens = np.cumsum(self.exp_lens)
        self.n_windows = np.sum(self.exp_lens)

    def __getitem__(self, idx):
        # mic idx
        pair_idx = idx % self.n_pairs
        mic1,mic2 = self.pair_idx_to_component_idx[pair_idx]

        # experiment name
        exp_i = np.argmax(self.c_exp_lens > idx)
        exp = self.exp_names[exp_i]

        #idx inside the experiment
        idx_in_exp = idx - (self.c_exp_lens[exp_i - 1] if exp_i > 0 else 0)
        window_idx = idx_in_exp//self.n_pairs

        sound = np.stack(self.sounds[exp][window_idx,[mic1,mic2]],axis=0)
        tdoa = (self.gt_toa[exp][window_idx, mic1] - self.gt_toa[exp][window_idx, mic2])*self.target_sampling_rate/self.speed_of_sound
        return sound, tdoa
        

    def get_sequence(self, exp, mics):
        if isinstance(exp, str):
            temp = [i for i,n in enumerate(self.exp_names) if exp == n]
            if len(temp) == 0:
                raise Exception("experiment name '" + exp + "' could not be found")
            exp_i = temp[0]
            exp_name = exp
        else:
            exp_i = exp
            exp_name = self.exp_names[exp]
        
        start_i = self.c_exp_lens[exp_i - 1] if exp_i > 0 else 0

        re_sound = self.sounds[exp_name][:,mics]
        re_tdoa = (self.gt_toa[exp_name][:,mics[0]] - self.gt_toa[exp_name][:,mics[1]])*self.target_sampling_rate/self.speed_of_sound
        re_time = self.times[exp_name]

        return re_sound, re_tdoa, re_time


    def __len__(self):
        return self.n_windows