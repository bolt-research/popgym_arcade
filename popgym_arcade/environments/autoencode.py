from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax import lax
import functools
from gymnax.environments import environment, spaces
from popgym_arcade.environments.draw_utils import (draw_heart,
                                            draw_spade,
                                            draw_club,
                                            draw_diamond,
                                            draw_number,
                                            draw_str,
                                            draw_sub_canvas)


@struct.dataclass
class EnvState:
    timestep: int
    cards: chex.Array
    score: int
    count: int
    default_action: int


@struct.dataclass
class EnvParams:
    pass


@jax.jit
def process_action(state: EnvState, action: int) -> Tuple[EnvState, bool]:
    new_default_action = jnp.where(
        action == 2,
        state.default_action + 1,
        jnp.where(
            action == 3,
            state.default_action - 1,
            state.default_action
        )
    )
    new_default_action = jnp.where(
        new_default_action < 0,
        3,
        jnp.where(
            new_default_action > 3,
            0,
            new_default_action
        )
    )
    return state.replace(default_action=new_default_action), action == 4


class AutoEncode(environment.Environment):
    """
    JAX compilable environment for AutoEncode.
    Source: https://github.com/proroklab/popgym/blob/master/popgym/envs/autoencode.py

    ### Description
    In the AutoEncode environment, the agent is presented with a sequence of cards, 
    each belonging to one of four suits: Club, Spade, Heart, or Diamond. The agent's 
    task is to recall and output the sequence of cards it saw, but in reverse order. 
    For example, if the agent sees the sequence [Club, Spade, Heart], 
    it should output [Heart, Spade, Club]. There are three difficulties: Easy, Medium,
    and Hard. In Easy, the agent is presented with a single deck of cards, in Medium,
    two decks, and in Hard, three decks. The agent is presented with the sequence of
    cards in the watch stage, and then must recall and output the sequence in the play
    stage. The agent receives a reward of 1.0 / (num_cards) for each correct card it
    outputs in the play stage.

    ### Action Space
    | Action | Description                         |
    |--------|-------------------------------------|
    | 0      | Up (No-op)                          |
    | 1      | Down (No-op)                        |
    | 2      | Left (Cycle options left)           |
    | 3      | Right (Cycle options right)         |
    | 4      | Confirm (Lock in current selection) |

    ### Observation Space
    The observation space consists of 256x256x3 and 192x192x3 image embeddings.
    There are 4 suits in each difficulties: Club, Spade, Heart, Diamond. The score
    is shown in the top middle of the image. The score will increase by 1 if the agent
    plays the correct card in play stage. In the watch stage, the agent will not receive
    ang reward, so the score will not increase. The current suit is shown in the bottom
    middle of the image.
    
    In MDP version, the agent can see the full sequence of history cards, shows in 
    192x192x3 image embeddings. 
    
    In POMDP version, the agent can only see the current card, shows in 256x256x3 
    image in the top left corner. 

    Agent can always see the score and the current suit in the image, which is shown
    in the top middle and top left of the image respectively.

    ### Reward
    - Reward Scale: 1.0 / (num_cards)
    In watch stage, the agent will not receive any reward.
    In play stage, the agent will receive reward scale if the agent's action is correct.

    ### Termination & Truncation
    - Termination: The episode terminates when the agent has played all cards
    - Truncation: The episode will be truncated after 140 steps + num_cards

    ### Args
    num_decks: The number of decks of cards to use. Easy: 1, Medium: 2, Hard: 3
    partial_obs: Whether to use POMDP version of the environment or not.
    max_steps_in_episode: The maximum number of steps in an episode.
    """

    color = {
        "red": jnp.array([1, 0, 0]),
        "dark_red": jnp.array([0.75, 0.10, 0.10]),
        "bright_red": jnp.array([1.0, 0.19, 0.28]),
        "black": jnp.array([0, 0, 0]),
        "white": jnp.array([1, 1, 1]),
        "metallic_gold": jnp.array([0.85, 0.65, 0.13]),
        "light_gray": jnp.array([0.96, 0.96, 0.96]),
        "light_blue": jnp.array([0.68, 0.85, 0.90]),
        "electric_blue": jnp.array([0.0, 0.45, 0.74]),
        "neon_pink": jnp.array([1.0, 0.41, 0.73]),
    }
    value_cards_pos = {
        "top_left": (0, 0),
        "bottom_right": (20, 40),
    }
    value_suit_pos = {
        "top_left": (0, 0),
        "bottom_right": (20, 40),
    }
    left_triangle_pos = {
        "top_left": (92, 224),
        "bottom_right": (112, 256),
    }
    current_suit_pos = {
        "top_left": (234, 0),
        "bottom_right": (254, 40),
    }
    right_triangle_pos = {
        "top_left": (152, 224),
        "bottom_right": (172, 256),
    }
    name_pos = {
        "top_left": (0, 256 - 25),
        "bottom_right": (256, 256),
    }
    score = {
        "top_left": (86, 2),
        "bottom_right": (171, 30),
    }

    def __init__(
            self,
            num_decks=1,
            partial_obs=False
    ):
        super().__init__()
        self.partial_obs = partial_obs
        self.num_suits = 4
        self.decksize = 26
        self.num_decks = num_decks
        self.canva_size = 256
        self.canva_color = self.color["light_blue"]
        self.large_canva = jnp.ones((self.canva_size, self.canva_size, 3)) * self.canva_color
        self.small_canva_size = 192
        self.small_canva = jnp.ones((self.small_canva_size, self.small_canva_size, 3)) * self.canva_color


        self.max_steps_in_episode = 140 + self.decksize * self.num_decks
        self.setup_render_templates()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def name(self) -> str:
        return "AutoEncode"

    def step_env(
            self,
            key: chex.PRNGKey,
            state: EnvState,
            action: int,
            params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step of the environment."""
        new_state, fire_action = process_action(state, action)
        num_cards = self.decksize * self.num_decks
        reward = 0

        fire_action = jnp.where(new_state.timestep <= num_cards, True, fire_action)

        reward_scale = 1.0 / (num_cards)

        terminated = jnp.logical_or(
            new_state.count >= num_cards,  # play all cards
            new_state.timestep >= self.max_steps_in_episode  # timelimit
        )
        play = new_state.timestep >= num_cards

        reward = jnp.where(
            fire_action,
            jnp.where(
                jnp.flip(new_state.cards, axis=0)[new_state.count] == new_state.default_action,
                reward_scale,
                0,
            ),
            0,
        )

        reward = jnp.where(
            play,
            reward,
            0,
        )

        new_state = EnvState(
            new_state.timestep + 1,
            new_state.cards,
            new_state.score + lax.cond(reward > 0, lambda _: 1, lambda _: 0, None),
            new_state.count + jnp.where(jnp.logical_and(fire_action, play), 1, 0),
            new_state.default_action,
        )
        obs = self.get_obs(new_state)

        return obs, new_state, reward, terminated, {}

    def reset_env(
            self,
            key: chex.PRNGKey,
            params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        cards = jnp.arange(self.decksize * self.num_decks) % self.num_suits
        cards = jax.random.permutation(key, cards)
        state = EnvState(
            timestep=0,
            cards=cards,
            score=0,
            count=0,
            default_action=0,
        )
        obs = self.get_obs(state)
        return obs, state

    def setup_render_templates(self):
        base_large = self.large_canva.copy()
        base_small = self.small_canva.copy()

        value_suit_top_left = self.value_suit_pos["top_left"]
        value_suit_bottom_right = self.value_suit_pos["bottom_right"]
        
        self.value_card_templates = jnp.stack([
            draw_heart(value_suit_top_left, value_suit_bottom_right, self.color["red"], base_large),
            draw_spade(value_suit_top_left, (value_suit_bottom_right[0], value_suit_bottom_right[1] - 6), self.color["black"], base_large),
            draw_club(value_suit_top_left, (value_suit_bottom_right[0], value_suit_bottom_right[1] - 6), self.color["black"], base_large),
            draw_diamond(value_suit_top_left, value_suit_bottom_right, self.color["red"], base_large)
        ])

        hist_positions = jnp.array([((i%9)*20, (i//9)*20) for i in range(self.decksize*self.num_decks)])
        
        hist_endings_red = hist_positions + jnp.array([12, 20])
        hist_endings_black = hist_positions + jnp.array([12, 14])
        
        vmap_draw = lambda fn: jax.vmap(fn, in_axes=(0, 0, None, None))
        self.history_card_templates = jnp.stack([
            vmap_draw(draw_heart)(hist_positions, hist_endings_red, self.color["red"], base_small),
            vmap_draw(draw_spade)(hist_positions, hist_endings_black, self.color["black"], base_small),
            vmap_draw(draw_club)(hist_positions, hist_endings_black, self.color["black"], base_small),
            vmap_draw(draw_diamond)(hist_positions, hist_endings_red, self.color["red"], base_small)
        ], axis=1)

    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnvState) -> chex.Array:
        large_canva = self.large_canva.copy()
        small_canva = self.small_canva.copy()

        valid_current_card = state.timestep < self.decksize * self.num_decks
        current_suit = jax.lax.select(
            valid_current_card,
            state.cards[state.timestep].astype(int),
            0
        )
        large_canva = jnp.where(
            valid_current_card,
            self.value_card_templates[current_suit],
            large_canva
        )

        def render_history(canvas):
            num_cards = self.decksize * self.num_decks
            valid_mask = jnp.arange(num_cards) < state.timestep

            card_indices = state.cards.astype(int)[:, None, None, None, None]
            
            selected = jnp.take_along_axis(
                self.history_card_templates,
                card_indices,
                axis=1
            ).squeeze(1)

            bg_color = self.small_canva[0, 0]
            valid_symbol = valid_mask[:, None, None] & jnp.any(selected != bg_color, axis=-1)
            
            card_priority = jnp.arange(num_cards)[:, None, None] * valid_symbol
            last_valid_idx = jnp.argmax(card_priority, axis=0)
            any_valid = jnp.any(valid_symbol, axis=0)
            
            h, w = jnp.indices((self.small_canva_size, self.small_canva_size))
            final_colors = selected[last_valid_idx, h, w]
            
            return jnp.where(any_valid[..., None], final_colors, canvas)
        
        small_canva = lax.cond(
            self.partial_obs,
            lambda: small_canva,
            lambda: render_history(small_canva)
        )

        a_pos = (self.current_suit_pos["top_left"], 
                self.current_suit_pos["bottom_right"])
        action_color = jnp.array([
            self.color["red"], 
            self.color["black"], 
            self.color["black"], 
            self.color["red"]
        ])[state.default_action]
        
        large_canva = jax.lax.switch(
            state.default_action,
            [
                lambda p0, p1, c, cnvs: draw_heart(p0, p1, c, cnvs),
                lambda p0, p1, c, cnvs: draw_spade(p0, p1, c, cnvs),
                lambda p0, p1, c, cnvs: draw_club(p0, p1, c, cnvs),
                lambda p0, p1, c, cnvs: draw_diamond(p0, p1, c, cnvs)
            ],
            a_pos[0], a_pos[1], action_color, large_canva
        )

        large_canva = draw_number(self.score["top_left"], self.score["bottom_right"],
                                self.color["bright_red"], large_canva, state.score)
        large_canva = draw_str(self.name_pos["top_left"], self.name_pos["bottom_right"],
                            self.color["neon_pink"], large_canva, self.name)
        
        return draw_sub_canvas(small_canva, large_canva)

    def get_obs(self, state: EnvState) -> chex.Array:
        """Returns observation from the state."""
        obs = self.render(state)
        return obs

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_suits)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.array(0, ), jnp.array(1, ), (256, 256, 3), dtype=jnp.float32)


class AutoEncodeEasy(AutoEncode):
    def __init__(self, partial_obs=False):
        super().__init__(num_decks=1, partial_obs=partial_obs)


class AutoEncodeMedium(AutoEncode):
    def __init__(self, partial_obs=False):
        super().__init__(num_decks=2, partial_obs=partial_obs)


class AutoEncodeHard(AutoEncode):
    def __init__(self, partial_obs=False):
        super().__init__(num_decks=3, partial_obs=partial_obs)
