import jax
import jax.numpy as jnp

import popgym_arcade


def test_continuous_cartpole_action_space():
    env, env_params = popgym_arcade.make("ContinuousCartPoleEasy")

    action_space = env.action_space(env_params)

    assert env.name == "ContPole"
    assert action_space.shape == (1,)
    assert action_space.dtype == jnp.float32
    assert action_space.low == -1.0
    assert action_space.high == 1.0
    assert env.num_actions == 1


def test_continuous_cartpole_force_is_continuous_and_clipped():
    env, env_params = popgym_arcade.make("ContinuousCartPoleEasy")

    assert env._action_to_force(jnp.array([-0.5]), env_params) == -5.0
    assert env._action_to_force(jnp.array([0.25]), env_params) == 2.5
    assert env._action_to_force(jnp.array([-2.0]), env_params) == -10.0
    assert env._action_to_force(jnp.array([2.0]), env_params) == 10.0


def test_continuous_cartpole_vectorized_step():
    env, env_params = popgym_arcade.make(
        "ContinuousCartPoleEasy", partial_obs=True, obs_size=128
    )
    n_envs = 2
    reset_keys = jax.random.split(jax.random.key(0), n_envs)
    step_keys = jax.random.split(jax.random.key(1), n_envs)
    actions = jnp.array([[-1.0], [1.0]], dtype=jnp.float32)

    observations, states = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(
        reset_keys, env_params
    )
    observations, states, rewards, dones, infos = jax.jit(
        jax.vmap(env.step, in_axes=(0, 0, 0, None))
    )(step_keys, states, actions, env_params)

    assert observations.shape == (n_envs, 128, 128, 3)
    assert rewards.shape == (n_envs,)
    assert dones.shape == (n_envs,)
    assert env.observation_space(env_params).contains(observations)
