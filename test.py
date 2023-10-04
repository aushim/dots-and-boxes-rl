import os
import glob
from self_play import SelfPlayWrapperEnv
from environments import DotsAndBoxesEnv
from sb3_contrib import MaskablePPO
from params import NUM_ROWS, NUM_COLUMNS, NUM_AGENTS, SEED


def test_run(
    agent_model_path=None,
    num_episodes=1000,
    use_trained_opponent=False,
    opponent_model_path_pattern=None,
    render=False,
    verbose=1,
):
    # Display gap between training and testing
    print()
    print()

    # Load the agent model
    if agent_model_path is None:
        print('No agent model path provided')
        return None

    model = None
    try:
        agent_policy = max(
            glob.glob(f"{agent_model_path}*.zip"), key=os.path.getctime
        )
        if agent_policy is not None:
            model = MaskablePPO.load(agent_policy)
            if model is not None:
                model.set_random_seed(42)
                print('Loaded agent model: {}'.format(agent_policy))
                print()
    except ValueError:
        print('Could not load agent model: {}'.format(agent_model_path))
        return None

    wins = 0
    total_score = 0
    env = SelfPlayWrapperEnv(env_create_func=DotsAndBoxesEnv,
                             render_mode=('human' if render else None),
                             primary_agent_index=0,
                             use_trained_opponent=use_trained_opponent,
                             opponent_model_path_pattern=opponent_model_path_pattern,
                             rows=NUM_ROWS,
                             columns=NUM_COLUMNS,
                             num_agents=NUM_AGENTS,
                             )
    env.reset(seed=SEED)
    print('Testing agent vs {}'.format(
        'trained opponent' if use_trained_opponent else 'random opponent'))
    print('============================================')
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            if render:
                env.render()
            # Choose action with highest probability
            action, _ = model.predict(obs, action_masks=env.action_masks())
            obs, reward, done, truncated, info = env.step(action.item())
            score += reward
        win = score >= 1
        total_score += score
        if win:
            wins += 1
        if verbose > 0:
            print('Episode:{} Score:{}'.format(episode + 1, score))
    win_rate = wins / num_episodes
    avg_score = total_score / num_episodes
    print('============================================')
    print('Average score: {}'.format(avg_score))
    print('Win rate: {}% ({}/{})'.format(win_rate * 100, wins, num_episodes))

    env.close()
