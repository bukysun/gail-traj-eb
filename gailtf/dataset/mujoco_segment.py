from gailtf.dataset.mujoco_traj import Traj_Dset, Mujoco_Traj_Dset
import numpy as np
from tqdm import tqdm
import pickle as pkl


class Mujoco_Segment_Dset(Mujoco_Traj_Dset):
    def __init__(self, segment_size = 0, expert_path = None, segments_path = None, ret_threshold = None, traj_limitation = np.inf, randomize = True, sentence_size = None):
        if segments_path is None:
            Mujoco_Traj_Dset.__init__(self, expert_path, ret_threshold = ret_threshold, traj_limitation = traj_limitation, randomize = randomize, sentence_size = sentence_size)
            self.segment_size = segment_size
            segments = []
            segments_len = []
            for traj, traj_len in tqdm(zip(self.traj_feats, self.lens)):
                traj_seg, traj_seg_len = self.traj2segment(traj[:traj_len,:], self.segment_size)
                seg = np.reshape(traj_seg, [traj_len*self.segment_size, segment_size, -1])
                seg_len = np.reshape(traj_seg_len, (traj_len * self.segment_size, -1))
                segments.append(seg)
                segments_len.append(seg_len)
            self.segments = np.concatenate(segments, 0)
            self.segments_len = np.concatenate(segments_len, 0)
        else:
            self.load_segments(segments_path)
        self.segments_dset = Traj_Dset(self.segments, self.segments_len, randomize)


    def get_next_seg_batch(self, batch_size):
        return self.segments_dset.get_next_batch(batch_size)

    def save_segments(self, f_path):
        with open(f_path, 'wb') as f:
            pkl.dump((self.segments, self.segments_len), f)

    def load_segments(self, f_path):
        with open(f_path, 'rb') as f:
            self.segments, self.segments_len = pkl.load(f)

    @staticmethod
    def traj2segment(traj, seg_size):
        assert len(traj.shape) == 2
        tl, fl = traj.shape
        segment = np.zeros((tl,seg_size, seg_size,fl))
        segment_len = np.zeros((tl, seg_size))
        for t in range(tl):
            trange = np.clip(t + np.arange(seg_size), 0, tl-1)
            for i in range(seg_size):
                segment[t, i, :(trange[i]+1-t),:] = traj[t:trange[i]+1,:]
                segment_len[t, i] = trange[i]+1-t 
        return segment, segment_len

def test(expert_path):
    dset = Mujoco_Segment_Dset(segments_path = "segments.expert.segment_length_6.pkl")
    #dset.save_segments("segments.expert.segment_length_6.pkl")
    res = dset.get_next_seg_batch(30)
    print(res.shape)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../baselines/ppo1/ppo.Hopper.0.00.pkl")
    args = parser.parse_args()
    test(args.expert_path)



