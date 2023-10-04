import time
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from environments import DotsAndBoxesEnv
from stable_baselines3.common.monitor import Monitor
from self_play import SelfPlayWrapperEnv
from callbacks import EvalCallback, UpdateOpponentOnRewardThreshold, StopTrainingOnNoModelImprovement
from params import ENV_NAME, NUM_ROWS, NUM_COLUMNS, NUM_AGENTS, SEED, TOTAL_TIMESTEPS, REWARD_THRESHOLD
from test import test_run

training_start_time = time.strftime('%Y%m%d-%H%M%S')
training_dir = './training/{}/{}x{}/{}-players/{}'.format(
    ENV_NAME, NUM_ROWS, NUM_COLUMNS, NUM_AGENTS, training_start_time)
models_dir = '{}/models'.format(training_dir)
logs_dir = '{}/logs'.format(training_dir)

primary_model_path = '{}/agent_final'.format(models_dir)
best_model_name = 'best_model'
opponent_model_path_pattern = '{}/{}'.format(models_dir, best_model_name)

env = Monitor(SelfPlayWrapperEnv(env_create_func=DotsAndBoxesEnv,
                                 primary_agent_index=0,
                                 use_trained_opponent=False,
                                 opponent_model_path_pattern=opponent_model_path_pattern,
                                 rows=NUM_ROWS,
                                 columns=NUM_COLUMNS,
                                 num_agents=NUM_AGENTS,
                                 ))
env.reset(seed=SEED)

stop_train_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=3, min_evals=5, verbose=1)
callback_on_best = UpdateOpponentOnRewardThreshold(
    reward_threshold=REWARD_THRESHOLD, verbose=1)
eval_callback = EvalCallback(
    env, best_model_save_path=models_dir, best_model_save_name=best_model_name,
    log_path=logs_dir, eval_freq=(TOTAL_TIMESTEPS // 10), n_eval_episodes=1000,
    deterministic=True, render=False, callback_after_eval=stop_train_callback,
    callback_on_new_best=callback_on_best, verbose=1)

# Train the agent
model = MaskablePPO(MaskableActorCriticPolicy, env,
                    verbose=1, tensorboard_log=logs_dir)
model.set_random_seed(SEED)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
model.save(primary_model_path)

env.close()

# Test the trained agent against random agent
test_run(agent_model_path=primary_model_path, num_episodes=1000,
         use_trained_opponent=False, opponent_model_path_pattern=opponent_model_path_pattern,
         render=False, verbose=0)

# Test the trained agent against itself
test_run(agent_model_path=primary_model_path, num_episodes=1000,
         use_trained_opponent=True, opponent_model_path_pattern=opponent_model_path_pattern,
         render=False, verbose=0)

# Watch the trained agent play against itself
test_run(agent_model_path=primary_model_path, num_episodes=5,
         use_trained_opponent=True, opponent_model_path_pattern=opponent_model_path_pattern,
         render=True, verbose=1)
