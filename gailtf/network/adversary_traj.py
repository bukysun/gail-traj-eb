from gailtf.baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow as tf
import tensorflow.contrib.layers as layers
from gailtf.baselines.common import tf_util as U
from gailtf.common.tf_util import *
import numpy as np
import ipdb

L2_REG = 1e-4

class TrajectoryClassifier(object):
    def __init__(self, env, hidden_size, sequence_size, attention_size, cell_type, entcoeff=0.001, lr_rate = 0.0, scope = "adversary"):
        self.scope = scope
        self.observation_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape
        self.num_observations = self.observation_shape[0]
        self.num_actions = self.action_shape[0]
        self.embedding_size = self.num_observations + self.num_actions
        self.hidden_size = hidden_size
        self.sequence_size = sequence_size
        self.attention_size = attention_size
        self.cell_type = cell_type
        self.build_ph()
        #Build graph
        generator_logits, self.rewards_op = self.build_graph(self.generator_traj_ph, self.generator_traj_seq_len, reuse = False)
        expert_logits, _ = self.build_graph(self.expert_traj_ph, self.expert_traj_seq_len, reuse = True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff*entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function([self.generator_traj_ph, self.generator_traj_seq_len, self.expert_traj_ph, self.expert_traj_seq_len, self.dropout_keep_prob], 
                                 self.losses + [U.flatgrad(self.total_loss, var_list)])
        # for test
        #self.check_values = U.function([self.generator_traj_ph, self.generator_traj_seq_len, self.expert_traj_ph, self.expert_traj_seq_len, self.dropout_keep_prob],[self.cvs, self.exp_cvs])
        
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


    def build_ph(self):
        self.generator_traj_ph = tf.placeholder(tf.float32, (None, self.sequence_size, self.embedding_size), name = "observation_traj")
        self.generator_traj_seq_len = tf.placeholder(tf.float32, (None,), name = "observation_seq_length")
        self.expert_traj_ph = tf.placeholder(tf.float32, (None, self.sequence_size, self.embedding_size), name = "expert_traj")
        self.expert_traj_seq_len = tf.placeholder(tf.float32, (None,), name = "expert_seq_length")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = 'dropout_keep_prob')

    def build_graph(self, trajs,trajs_len, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            #input normalize
            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape = self.observation_shape)
            obs = (trajs[:,:,:self.num_observations] - self.obs_rms.mean) / self.obs_rms.std
            feats = tf.concat((obs, trajs[:,:,self.num_observations:]), 2)
            #feats = trajs

            with tf.variable_scope("rnn"):
                cell = self._get_cell(self.hidden_size,self.cell_type, reuse)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = self.dropout_keep_prob)
                outputs, _ = tf.nn.dynamic_rnn(cell = cell, inputs = feats, sequence_length = trajs_len, dtype = tf.float32)
                with tf.variable_scope('attention') as scope:
                    attn_outputs, weighted_eb = self.attention(outputs, self.attention_size, scope)
                logits = self.shared_fc_layer(attn_outputs, reuse = False)
                rewards = self.shared_fc_layer(weighted_eb, reuse = True)
                #check_values = (outputs, attn_outputs, weighted_eb)
        return logits, rewards#, check_values

    def shared_fc_layer(self, inputs, scope = 'fully_connected', reuse = False):
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            outputs = tf.contrib.layers.fully_connected(inputs, 1, activation_fn = tf.identity)
        return outputs

    def attention(self, inputs, size, scope):
        with tf.variable_scope(scope or "attention") as scope:
            attention_context_vector = tf.get_variable(name = "attention_context_vector", shape = [size], regularizer = layers.l2_regularizer(scale = L2_REG), dtype = tf.float32)
            input_projection = layers.fully_connected(inputs, size, activation_fn = tf.tanh, weights_regularizer=layers.l2_regularizer(scale=L2_REG))
            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
            attention_weights = tf.nn.softmax(vector_attn, dim = 1)
            weighted_projection = tf.multiply(inputs, attention_weights)
            outputs = tf.reduce_sum(weighted_projection, axis = 1)
        
        return outputs, weighted_projection

    @staticmethod
    def _get_cell(hidden_size, cell_type = 'lstm', reuse = False):
        if cell_type == "vanilla":
            return tf.contrib.rnn.BasicRNNCell(hidden_size, reuse = reuse)
        elif cell_type == "lstm":
            return tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse = reuse)
        elif cell_type == "gru":
            return tf.contrib.rnn.GRUCell(hidden_size, reuse = reuse)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None

    def get_reward(self, trajs, trajs_len, dropout_keep_prob = 1.0):
        sess = U.get_session()
        if len(trajs.shape) == 2:
            trajs = np.expand_dims(trajs, 0)
        if len(np.shape(trajs_len)) == 0:
            trajs_len = np.expand_dims(trajs_len, 0)
        feed_dict = {self.generator_traj_ph:trajs, self.generator_traj_seq_len:trajs_len, self.dropout_keep_prob:dropout_keep_prob}
        rewards = sess.run(self.rewards_op, feed_dict)
        return rewards

def test(expert_path,sequence_size = 1000,attention_size = 30, hidden_size = 30, env_id = 'Hopper-v1', cell_type = 'lstm'):
    from gailtf.dataset.mujoco_traj import Mujoco_Traj_Dset
    import gym
    U.make_session(num_cpu = 2).__enter__()
    dset = Mujoco_Traj_Dset(expert_path)
    env = gym.make(env_id)
    t1, tl1 = dset.get_next_traj_batch(10)
    t2, tl2 = dset.get_next_traj_batch(10)
    discriminator = TrajectoryClassifier(env, hidden_size, sequence_size, attention_size, cell_type)
    U.initialize()

    *losses, g = discriminator.lossandgrad(t1, tl1, t2, tl2, 0.5)
    rs1 = discriminator.get_rewards(t1,tl1)
    #cv1,cv2 = discriminator.check_values(t1,tl1,t2,tl2,0.5)
    print(rs1.shape)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../baselines/ppo1/ppo.Hopper.0.00.pkl")
    args = parser.parse_args()
    test(args.expert_path)








