from gailtf.baselines.common import explained_variance, zipsame, dataset, Dataset, fmt_row
from gailtf.baselines import logger
import gailtf.baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time, os
from gailtf.baselines.common import colorize
from mpi4py import MPI
from collections import deque
from gailtf.baselines.common.mpi_adam import MpiAdam
from gailtf.baselines.common.cg import cg
from contextlib import contextmanager
from gailtf.common.statistics import stats
import ipdb

def traj_segment_generator(pi, env, discriminator, episodes_num, stochastic, seq_length):
    #Initialize state variable
    ep_cnt = 0
    ob = env.reset()
    ac = env.action_space.sample()
    new = True
    # For single episode
    cur_ep_obs = []
    cur_ep_acs = []
    cur_ep_rews = []
    cur_ep_true_rews = []
    cur_ep_vpreds = []
    cur_ep_news = []
    cur_ep_prevacs = []
    # For single call
    ep_obs = []
    ep_acs = []
    ep_rews = []
    ep_true_rews = []
    ep_vpreds =[]
    ep_news = []
    ep_prevacs = []
    ep_true_rets = []
    ep_rets = []
    ep_lens = []
    ep_trajs = []



    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        if ep_cnt > 0 and new and ep_cnt % episodes_num == 0:
            # concatenate multiple trajectories into one
            obs = np.concatenate(ep_obs, 0)
            acs = np.concatenate(ep_acs, 0)
            rews = np.concatenate(ep_rews, 0)
            vpreds = np.concatenate(ep_vpreds, 0)
            news = np.concatenate(ep_news, 0)
            prevacs = np.concatenate(ep_prevacs, 0)
            #output for one batch
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                    "ac": acs, "prevacs": prevacs, "nextvpred": vpred * (1 - new),
                    "ep_obs":ep_obs, "ep_acs":ep_acs,"ep_trajs":np.array(ep_trajs), "ep_rets": ep_rets, 
                    "ep_lens":np.array(ep_lens), "ep_true_rets":ep_true_rets}
            
            _, vpred = pi.act(stochastic, ob)
            #clear episodes list
            ep_obs = []
            ep_acs = []
            ep_rews = []
            ep_true_rews = []
            ep_vpreds =[]
            ep_news = []
            ep_prevacs = []
            ep_true_rets = []
            ep_rets = []
            ep_lens = []
            ep_trajs = []
        cur_ep_obs.append(ob)
        cur_ep_vpreds.append(vpred)
        cur_ep_news.append(new)
        cur_ep_acs.append(ac)
        cur_ep_prevacs.append(prevac)

        ob, true_rew, new, _ = env.step(ac)
        cur_ep_true_rews.append(true_rew)
        
        if new:
            #calculate rewards
            cur_ep_rews, cur_ep_traj = get_cur_ep_rewards(discriminator, np.array(cur_ep_obs), np.array(cur_ep_acs), seq_length)
            # add current trajectory to episodes list
            ep_obs.append(np.array(cur_ep_obs))
            ep_acs.append(np.array(cur_ep_acs))
            ep_rews.append(cur_ep_rews)
            ep_true_rews.append(cur_ep_true_rews)
            ep_vpreds.append(cur_ep_vpreds)
            ep_news.append(cur_ep_news)
            ep_prevacs.append(cur_ep_prevacs)
            ep_true_rets.append(np.sum(cur_ep_true_rews))
            ep_rets.append(np.sum(cur_ep_rews))
            ep_lens.append(len(cur_ep_obs))
            ep_trajs.append(cur_ep_traj)
            #clear current trajectory list
            cur_ep_obs = []
            cur_ep_acs = []
            cur_ep_rews = []
            cur_ep_true_rews = []
            cur_ep_vpreds = []
            cur_ep_news = []
            cur_ep_prevacs = []
            ob = env.reset()
            ep_cnt += 1

def get_cur_ep_rewards(discriminator, obs, acs, seq_length):
    assert obs.shape[0] == acs.shape[0]
    traj_len = obs.shape[0]
    traj = np.vstack((np.hstack((obs, acs)), np.zeros((seq_length-traj_len, obs.shape[1]+acs.shape[1]))))
    assert traj.shape[0] == seq_length
    rews = discriminator.get_reward(traj, traj_len)
    rews = np.squeeze(rews)
    rews = rews[:traj_len]
    return rews, traj


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, policy_func, discriminator, expert_dataset,
        pretrained, pretrained_weight, *,
        g_step, d_step,
        episodes_per_batch, # what to train on
        dropout_keep_prob, sequence_size, #rnn parameters
        max_kl, cg_iters,
        gamma, lam, # advantage estimation
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4, d_stepsize=3e-4,
        vf_iters =3,
        max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        save_per_iter=100, ckpt_dir=None, log_dir=None, 
        load_model_path=None, task_name=None
        ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3) 
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=(pretrained_weight!=None))
    oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    entbonus = entcoeff * meanent

    vferr = U.mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = U.mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    d_adam = MpiAdam(discriminator.get_trainable_variables())
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([U.sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield
    
    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    writer = U.FileWriter(log_dir)
    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    d_adam.sync()
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, discriminator, episodes_per_batch, stochastic=True, seq_length = sequence_size)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=40)

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

    g_loss_stats = stats(loss_names)
    d_loss_stats = stats(discriminator.loss_name)
    ep_stats = stats(["True_rewards", "Rewards", "Episode_length"])
    # if provide pretrained weight
    if pretrained_weight is not None:
        U.load_state(pretrained_weight, var_list=pi.get_variables())
    # if provieded model path
    if load_model_path is not None:
        U.load_state(load_model_path)

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        
        # Save model
        if iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            U.save_state(os.path.join(ckpt_dir, task_name), counter=iters_so_far)
        
        logger.log("********** Iteration %i ************"%iters_so_far)

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p
        # ------------------ Update G ------------------
        logger.log("Optimizing Policy...")
        for _ in range(g_step):
            with timed("sampling"):
                seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)
            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

            if hasattr(pi, "ob_rms"):pi.ob_rms.update(ob)
            
            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]

            assign_old_eq_new() # set old parameter values to new parameter values
            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / max_kl)
                # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > max_kl * 1.5:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    set_from_flat(thbefore)
                if nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
            with timed("vf"):
                for _ in range(vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                    include_final_partial_batch=False, batch_size=128):
                        if hasattr(pi, "ob_rms"): pi.ob_rms.update(mbob) # update running mean/std for policy
                        g = allmean(compute_vflossandgrad(mbob, mbret))
                        vfadam.update(g, vf_stepsize)
        
        g_losses = meanlosses
        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret)) 
        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, discriminator.loss_name))
        traj_gen, traj_len_gen = seg["ep_trajs"], seg["ep_lens"]
        #traj_expert, traj_len_expert = expert_dataset.get_next_traj_batch()
        batch_size = len(traj_gen) // d_step
        d_losses = [] # list of tuples, each of which gives the loss for a minibatch
        for traj_batch, traj_len_batch in dataset.iterbatches((traj_gen, traj_len_gen),
                    include_final_partial_batch = False, batch_size = batch_size):
            traj_expert, traj_len_expert = expert_dataset.get_next_traj_batch(len(traj_batch))
            # update running mean/std for discriminator
            ob_batch, _ = traj2trans(traj_batch, traj_len_batch, ob_space.shape[0])
            ob_expert, _ = traj2trans(traj_expert, traj_len_expert, ob_space.shape[0])
            if hasattr(discriminator, "obs_rms"): discriminator.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = discriminator.lossandgrad(traj_batch, traj_len_batch, traj_expert, traj_len_expert, dropout_keep_prob)
            d_adam.update(allmean(g), d_stepsize)
            d_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))


        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rewbuffer.extend(true_rets)
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()
            g_loss_stats.add_all_summary(writer, g_losses, iters_so_far)
            d_loss_stats.add_all_summary(writer, np.mean(d_losses, axis=0), iters_so_far)
            ep_stats.add_all_summary(writer, [np.mean(true_rewbuffer), np.mean(rewbuffer),
                           np.mean(lenbuffer)], iters_so_far)




def traj2trans(trajs, traj_lens, num_ob):
    obs = []
    acs = []
    for t, tl in zip(trajs, traj_lens):
        ob = t[:tl, :num_ob]
        ac = t[:tl, num_ob:]
        obs.append(ob)
        acs.append(ac)
    return np.concatenate(obs, 0), np.concatenate(acs, 0)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
    



def test_seg_gen(sequence_size = 1000,attention_size = 30, hidden_size = 30, env_id = 'Hopper-v1', cell_type = 'lstm'):
    from gailtf.baselines.ppo1 import mlp_policy
    from gailtf.network.adversary_traj import TrajectoryClassifier
    import gym
    env = gym.make("Hopper-v1")
    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            reuse=reuse, hid_size=64, num_hid_layers=2)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn('pi', ob_space, ac_space)
    discriminator = TrajectoryClassifier(env, hidden_size, sequence_size, attention_size, cell_type)
    U.make_session(num_cpu = 2).__enter__()
    U.initialize()
    seg_gen = traj_segment_generator(pi, env, discriminator, 10, True, sequence_size)
    for i in range(10):
        seg = seg_gen.__next__()
        ob, ac = traj2trans(seg["ep_trajs"], seg["ep_lens"], ob_space.shape[0])
        add_vtarg_and_adv(seg, gamma=0.995, lam=0.97)
        print(seg['adv'].shape, seg['tdlamret'].shape, seg['ob'].shape, seg['nextvpred'])
    

if __name__ == "__main__":
    test_seg_gen()



