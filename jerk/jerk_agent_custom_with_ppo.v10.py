#!/usr/bin/env python

"""
A scripted agent called "Just Enough Retained Knowledge".
"""

import random
import copy

import gym
import numpy as np

import gym_remote.client as grc
import gym_remote.exceptions as gre

#ppo requirements
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines import logger

from sonic_util_train import AllowBacktracking, SonicDiscretizer, RewardScaler, FrameStack, WarpFrame, make_env
import traceback
import threading
import joblib
import time

#EMA_RATE = 0.2
EXPLOIT_BIAS = 0.0 #0.25 # now redundant with the additional checks below
TOTAL_TIMESTEPS = int(1e6)
MAX_SCORE=10000
MUTATION_DAMPEN = 0.3

BUTTONS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
COMBOS = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
           ['DOWN', 'B'], ['B']]
valid_actions = []
for action in COMBOS:
    arr = np.array([False] * 12)
    for button in action:
        arr[BUTTONS.index(button)] = True
    valid_actions.append(arr)
ACTIONS = valid_actions


class JerkAgent(threading.Thread):
    def __init__(self, env, solutions = []):
        threading.Thread.__init__(self)
        self.env = TrackedEnv(env)
        self.solutions = solutions

    def run(self):
        self.train()

    def should_use_history(self, reward, env):
        reward_percentage = reward / MAX_SCORE

        the_end_is_nigh = (EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS) ** 3

        return random.random() < np.mean([reward_percentage, the_end_is_nigh])

    def best_solution(self):
        best_pair = sorted(self.solutions, key=lambda x: np.mean(x[0]))[-1]
        reward = np.mean(best_pair[0])

        return best_pair, reward

    def train(self):
        """Run JERK on the attached environment."""
        new_ep = True
        best_reward = 0.
        keep_percentage = 0.6
        best_pair = None

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True # pylint: disable=E1101
        with tf.Session(config=config):
            self.session = tf.get_default_session()
            self.model = policies.CnnPolicy(
                self.session,
                self.env.observation_space,
                self.env.action_space,
                1,
                1)
            self.a0 = self.model.pd.sample()
            params = tf.trainable_variables()
            print('params', params)
            load_path = '/root/compo/saved_weights.joblib'
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            self.session.run(restores)

            print('model created')

            while True:
                if new_ep:
                    if len(self.solutions) > 0:
                        best_pair, best_reward = self.best_solution()

                    if self.solutions and self.should_use_history(best_reward, self.env):
                        new_rew, last_reward_index = self.exploit(self.env, best_pair[1])
                        best_pair[0].append(new_rew)
                        if (new_rew / best_reward > keep_percentage) and len(self.env.best_sequence()) != len(best_pair[1]):
                            self.solutions.append(([max(self.env.reward_history)], self.env.best_sequence()[0:last_reward_index]))
                        print('replayed best with reward %f' % (new_rew * 100))
                        continue
                    elif best_pair:
                        mutation_rate = (1 - (best_reward / MAX_SCORE)) * MUTATION_DAMPEN
                        mutated = self.mutate(best_pair[1], mutation_rate)
                        new_rew, last_reward_index = self.exploit(self.env, mutated)
                        print('mutated solution rewarded %f vs %f' % ((new_rew * 100), (best_reward * 100)))
                        if (new_rew / best_reward > keep_percentage):
                            self.solutions.append(([max(self.env.reward_history)], self.env.best_sequence()[0:last_reward_index]))
                        continue
                    else:
                        self.env.reset()
                        new_ep = False
                rew, new_ep = self.move(self.env, 100)
                if not new_ep and rew <= 0:
                    print('backtracking due to negative reward: %f' % (rew * 100))
                    _, new_ep = self.move(self.env, 70, left=True)
                if new_ep:
                    self.solutions.append(([max(self.env.reward_history)], self.env.best_sequence()))


    def mutate(self, sequence, mutation_rate):
        mutated = copy.deepcopy(sequence)
        sequence_length = len(sequence)
        mutation_count = 0

        #mutation_start_index = min(sequence_length, random.randint(100, 2000))
        if random.random() < sequence_length / 100.:
            deletion_index = random.randint(0, sequence_length - 1)
            deletion_length = random.randint(0, sequence_length // 5)
            del mutated[deletion_index:(deletion_index + deletion_length)]
            print('excised %d of %d actions' % (deletion_length, sequence_length))

        mutation_start_index = len(mutated)

        for i, action in reversed(list(enumerate(sequence[0:mutation_start_index]))):
            #percent_distance = i + 1 / sequence_length
            exponent = -(mutation_start_index - i - 1) / 1e2
            if random.random() < np.exp(exponent) * mutation_rate:
                mutated = mutated[0:i]
                print('trimmed %d of %d actions' % (mutation_start_index - len(mutated), sequence_length))
                return mutated

                #mutated[i] = random.choice(ACTIONS)
                #mutation_count += 1
        print('mutated %d out of %d actions' % (mutation_count, sequence_length))

        return mutated

    def random_next_step(self, left=False, jump_prob=1.0 / 10.0, jump_repeat=4, jumping_steps_left=0):
        action = np.zeros((12,), dtype=np.bool)
        action[6] = left
        action[7] = not left
        if jumping_steps_left > 0:
            action[0] = True
            jumping_steps_left -= 1
        else:
            if random.random() < jump_prob:
                jumping_steps_left = jump_repeat - 1
                action[0] = True

        return action, jumping_steps_left

    def move(self, env, num_steps, left=False, jump_prob=1.0 / 10.0, jump_repeat=4):
        """
        Move right or left for a certain number of steps,
        jumping periodically.
        """
        #start_time = time.clock()
        total_rew = 0.0
        done = False
        steps_taken = 0
        jumping_steps_left = 0
        random_prob = 0.3
        use_model = random.random() > random_prob
        times = {}
        while not done and steps_taken < num_steps:
            if self.model and len(self.env.obs_history) > 0 and not left and use_model:
                #print('sample', time.clock() - start_time)
                ob = [self.env.obs_history[-1]]
                #print('fetched observation', time.clock() - start_time)
                actions = self.session.run([self.a0], {self.model.X: ob})
                #print('action', time.clock() - start_time)
                #for action in actions[0]:
                _, rew, done, _ = env.step(actions[0][0])
                #print('step', time.clock() - start_time)

            else:
                action, jumping_steps_left = self.random_next_step(left, jump_prob, jump_repeat, jumping_steps_left)
                _, rew, done, _ = env.step(action)

            total_rew += rew
            steps_taken += 1
            if done:
                break
        #print('time to move {} steps'.format(steps_taken), time.clock() - start_time)
        return total_rew, done

    def exploit(self, env, sequence):
        """
        Replay an action sequence; pad with NOPs if needed.

        Returns the final cumulative reward.
        """
        env.reset()
        done = False
        idx = 0
        total_reward = 0
        jumping_steps_left = 0
        left = False
        last_reward_index = 0
        while not done:
            if idx >= len(sequence) or idx - last_reward_index > 100:
                while not done:
                    steps = 100
                    reward, done = self.move(env, steps, left)
                    idx += steps
                    if left:
                        left = False
                    if reward == 0:
                        left = True
                    else:
                        last_reward_index = idx
            else:
                action = sequence[idx]
                _, reward, done, info = env.step(action)
                total_reward += reward
                if reward > 0:
                    last_reward_index = idx

            #_, _, done, _ = env.step(action)
            idx += 1
        return env.total_reward, last_reward_index

def main():
    """Run JERK on the attached environment."""
    env = grc.RemoteEnv('tmp/sock')
    env = SonicDiscretizer(env)
    env = RewardScaler(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)

    print('action space')
    print(env.action_space)
    print('observation space')
    print(env.observation_space)

    agent = JerkAgent(env)
    agent.train()

    return

class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        self.action_history = []
        self.reward_history = []
        self.obs_history = []
        self.total_reward = 0
        self.total_steps_ever = 0

    def best_sequence(self):
        """
        Get the prefix of the trajectory with the best
        cumulative reward.
        """
        max_cumulative = max(self.reward_history)
        for i, rew in enumerate(self.reward_history):
            if rew == max_cumulative:
                return self.action_history[:i+1]
        raise RuntimeError('unreachable')

    # pylint: disable=E0202
    def reset(self, **kwargs):
        self.action_history = []
        self.reward_history = []
        self.obs_history = []
        self.total_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        self.obs_history.append(obs)
        return obs, rew, done, info

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
        print(exc.args)
        traceback.print_exc()
