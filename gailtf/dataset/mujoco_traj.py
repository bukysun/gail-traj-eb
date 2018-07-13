from gailtf.baselines import logger
import pickle as pkl
import numpy as np
from tqdm import tqdm


class Traj_Dset(object):
    def __init__(self, inputs, inputs_len, randomize):
        self.inputs = inputs
        self.inputs_len = inputs_len
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()
       
    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.inputs_len = self.inputs_len[idx]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.inputs_len
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        inputs_len = self.inputs_len[self.pointer:end]
        self.pointer = end
        return inputs, inputs_len

class Mujoco_Traj_Dset(object):
    def __init__(self, expert_path, ret_threshold = None, traj_limitation = np.inf, randomize = True, sentence_size = None):
        with open(expert_path, "rb") as f:
            traj_data = pkl.load(f)
        obs = []
        acs = []
        rets = []
        lens = []
        for traj in tqdm(traj_data):
            if ret_threshold is not None and traj["ep_ret"] < ret_threshold:
                pass
            if len(rets) >= traj_limitation:
                break
            rets.append(traj["ep_ret"])
            lens.append(len(traj["ob"]))
            obs.append(traj["ob"])
            acs.append(traj["ac"])
        self.num_traj = len(rets)
        self.avg_ret = sum(rets)/len(rets)
        self.avg_len = sum(lens)/len(lens)
        self.rets = np.array(rets)
        self.lens = np.array(lens)
        if sentence_size is None:
            sentence_size = np.max(self.lens)
        self.sentence_size = sentence_size
        self.randomize = randomize

        self.traj_obs = np.array([ob for ob in obs])
        self.traj_acs = np.array([ac for ac in acs])
        self.traj_feats = []
        for ob, ac in zip(obs, acs):
            assert len(ob) == len(ac)
            tmp = np.hstack((ob,ac))
            tmp = np.vstack((tmp, np.zeros((self.sentence_size-tmp.shape[0], tmp.shape[1]))))
            self.traj_feats.append(tmp)
        self.traj_feats = np.array(self.traj_feats)
        self.dset = Traj_Dset(self.traj_feats,self.lens, self.randomize)
        self.log_info() 
        
    def log_info(self):
        logger.log("Total trajectories: %d"%self.num_traj)
        logger.log("Average episode length: %f"%self.avg_len)
        logger.log("Average returns: %f"%self.avg_ret)
    
    def get_next_traj_batch(self, batch_size):
        return self.dset.get_next_batch(batch_size)

def test(expert_path):
    dset = Mujoco_Traj_Dset(expert_path)
    s, sl = dset.get_next_traj_batch(10)
    print(sl)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../baselines/ppo1/ppo.Hopper.0.00.pkl")
    args = parser.parse_args()
    test(args.expert_path)



