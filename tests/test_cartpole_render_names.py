import popgym_arcade


def test_cartpole_render_names():
    cartpole, _ = popgym_arcade.make("CartPoleEasy")
    noisy_cartpole, _ = popgym_arcade.make("NoisyCartPoleEasy")
    continuous_cartpole, _ = popgym_arcade.make("ContinuousCartPoleEasy")

    assert cartpole.name == "CartPole"
    assert noisy_cartpole.name == "NoisyPole"
    assert continuous_cartpole.name == "ContPole"
