"""Shared construction of SB3 agents and vectorised envs."""
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .env import B601T1Env

PPO_POLICY_KWARGS = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh)
SAC_POLICY_KWARGS = dict(net_arch=[256, 256])


def make_vec_env(n_envs: int, seed: int, monitor_dir=None, env_kwargs=None, subproc=True):
    env_kwargs = env_kwargs or {}

    def thunk(rank):
        def _f():
            env = B601T1Env(**env_kwargs)
            fn = None if monitor_dir is None else str(monitor_dir / f"env{rank:02d}")
            env = Monitor(env, fn, info_keywords=("is_success",))
            env.reset(seed=seed * 1000 + rank)
            return env
        return _f

    cls = SubprocVecEnv if (subproc and n_envs > 1) else DummyVecEnv
    return cls([thunk(i) for i in range(n_envs)])


def make_ppo(venv, seed: int, lr=3e-4, log_std_init=0.0, n_steps=256, batch_size=1024,
             n_epochs=10, ent_coef=0.0, device="cpu", tb=None):
    kw = dict(PPO_POLICY_KWARGS, log_std_init=log_std_init)
    return PPO("MlpPolicy", venv, learning_rate=lr, n_steps=n_steps, batch_size=batch_size,
               n_epochs=n_epochs, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=ent_coef,
               vf_coef=0.5, max_grad_norm=0.5, policy_kwargs=kw, seed=seed, device=device,
               verbose=0, tensorboard_log=tb)


def make_sac(venv, seed: int, lr=3e-4, device="cuda", buffer_size=1_000_000, learning_starts=10_000,
             batch_size=512, train_freq=1, gradient_steps=1):
    return SAC("MlpPolicy", venv, learning_rate=lr, buffer_size=buffer_size,
               learning_starts=learning_starts, batch_size=batch_size, gamma=0.99, tau=0.005,
               train_freq=train_freq, gradient_steps=gradient_steps, policy_kwargs=SAC_POLICY_KWARGS,
               seed=seed, device=device, verbose=0)


def actor_parameters(ppo_model):
    p = ppo_model.policy
    return list(p.mlp_extractor.policy_net.parameters()) + list(p.action_net.parameters()) + [p.log_std]
