#!/usr/bin/env python3

import argparse

import jax
import numpy as np
import pygame
from gymnax.environments import spaces

import popgym_arcade
import popgym_arcade.registration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env", help="The env you want to play, for example BattleShipEasy"
    )
    parser.add_argument(
        "--partial", "-p", help="Play the POMDP variant, else MDP", action="store_true"
    )
    parser.add_argument(
        "--obs-size",
        "-o",
        help="Pixel size of observations, can be 128 or 256",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    assert (
        args.env in popgym_arcade.registration.REGISTERED_ENVIRONMENTS
    ), f"Invalid env: {args.env}, must be in {popgym_arcade.registration.REGISTERED_ENVIRONMENTS}"
    assert args.obs_size in [
        128,
        256,
    ], f"Invalid obs-size: {args.obs_size}, must be in [128, 256]"
    return args


def to_surf(arr):
    # Convert jax arry to pygame surface
    return np.transpose(arr, (1, 0, 2))


def play(args):
    # Create env env variant
    env, env_params = popgym_arcade.make(
        args.env, partial_obs=args.partial, obs_size=args.obs_size
    )

    # Vectorize and compile the env
    env_reset = jax.jit(env.reset)
    env_step = jax.jit(env.step)

    # Initialize environment
    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    observation, env_state = env_reset(reset_key, env_params)
    done = False
    action_space = env.action_space(env_params)
    continuous_actions = isinstance(action_space, spaces.Box)

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((args.obs_size, args.obs_size))
    clock = pygame.time.Clock()
    running = True

    # Convert numpy array to Pygame surface
    print(observation.dtype)
    surface = pygame.surfarray.make_surface(to_surf(observation))

    # Discrete action mappings
    action_meanings = {
        pygame.K_UP: 0,
        pygame.K_DOWN: 1,
        pygame.K_LEFT: 2,
        pygame.K_RIGHT: 3,
        pygame.K_SPACE: 4,
    }
    if continuous_actions:
        print("Controls: hold left/right to apply continuous force; release for zero")
    else:
        print("Controls: up, down, left, right, spacebar")

    while running:
        # Handle events
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif (
                not continuous_actions
                and event.type == pygame.KEYDOWN
                and event.key in action_meanings
            ):
                action = action_meanings[event.key]

        if continuous_actions:
            keys = pygame.key.get_pressed()
            action = np.zeros(action_space.shape, dtype=np.float32)
            if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
                action.flat[0] = -1.0
            elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
                action.flat[0] = 1.0

        # Render to screen
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Take action if key pressed
        if action is not None:
            key, step_key = jax.random.split(key)
            observation, env_state, reward, done, info = env_step(
                step_key, env_state, action, env_params
            )
            surface = pygame.surfarray.make_surface(to_surf(observation))

            # Render to screen
            if done:
                print("Game over")
                break

        clock.tick(30)  # Control FPS

    pygame.quit()


def main():
    args = parse_args()
    play(args)


if __name__ == "__main__":
    main()
