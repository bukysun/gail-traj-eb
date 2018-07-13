export GAILTF=/export/scratch/wuhui/workspace/gail-traj-eb
export ENV_ID="Hopper-v1"
export BASELINES_PATH=$GAILTF/gailtf/baselines/ppo1 # use gailtf/baselines/trpo_mpi for TRPO
export SAMPLE_STOCHASTIC="False"            # use True for stochastic sampling
export STOCHASTIC_POLICY="False"            # use True for a stochastic policy
export PYTHONPATH=$GAILTF:$PYTHONPATH       # as mentioned below
cd $GAILTF
export PATH_TO_CKPT=$GAILTF/checkpoint/ppo.Hopper.0.00/ppo.Hopper.0.00-400
export PICKLE_PATH=$GAILTF/stochastic.ppo.Hopper.0.00.pkl
export PATH_TO_GAILEV=$GAILTF/checkpoint/trpo_gail.Hopper.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001/trpo_gail.Hopper.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001-4900
export PATH_TO_BCEV=$GAILTF/checkpoint/behavior_cloning.Hopper/behavior_cloning.Hopper-9000
