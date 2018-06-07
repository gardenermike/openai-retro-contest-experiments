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
import os
import subprocess
import pickle
from annoy import AnnoyIndex

#from anyrl.rollouts import PrioritizedReplayBuffer

#EMA_RATE = 0.2
EXPLOIT_BIAS = 0.0 #0.25 # now redundant with the additional checks below
TOTAL_TIMESTEPS = int(1e6)
MAX_SCORE=10000
MUTATION_DAMPEN = 0.2

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
        #self.history = pickle.load(open('./history.pkl', 'rb'))
        self.history = []
        self.annoy_index = None #AnnoyIndex(512)
        #self.annoy_index.load('./test.ann')
        self.recorded_episode_count = 0
        #self.replay_buffer = PrioritizedReplayBuffer(1000000, 0.5, 0.4, epsilon=0.1)

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
            #print('params', params)
            #for i in tf.get_default_graph().get_operations():
            #    print(i.name)
            #self.output_layer = tf.get_default_graph().get_tensor_by_name('model/fc1/add:0')
            self.output_layer = tf.get_default_graph().get_tensor_by_name('model/Relu_3:0')
            #load_path = '/root/compo/saved_weights.joblib'
            load_path = './saved_weights.joblib'
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
                            self.record_history()
                        print('replayed best with reward %f' % (new_rew * 100))
                        continue
                    elif best_pair:
                        mutation_rate = (1 - (best_reward / MAX_SCORE)) * MUTATION_DAMPEN
                        mutated = self.mutate(best_pair[1], mutation_rate)
                        new_rew, last_reward_index = self.exploit(self.env, mutated)
                        print('mutated solution rewarded %f vs %f' % ((new_rew * 100), (best_reward * 100)))
                        if (new_rew / best_reward > keep_percentage):
                            self.solutions.append(([max(self.env.reward_history)], self.env.best_sequence()[0:last_reward_index]))
                            self.record_history()
                        continue
                    else:
                        self.env.reset()
                        new_ep = False
                rew, new_ep = self.move(self.env, 100)
                if not new_ep and rew <= 0:
                    print('backtracking due to negative reward: %f' % (rew * 100))
                    _, new_ep = self.move(self.env, 70, left=True)
                if new_ep:
                    print('Episode rewarded %f vs %f' % ((rew * 100), (best_reward * 100)))
                    self.record_history()
                    self.solutions.append(([max(self.env.reward_history)], self.env.best_sequence()))

    def record_history(self):
        self.recorded_episode_count += 1
        op = self.output_layer

        embeddings = []

        for i in range(len(self.env.reward_history)):
            obs = self.env.obs_history[i] #(i * batch_size):((i + 1) * batch_size)]
            embedding = self.session.run([op], {self.model.X: [obs]})[0].reshape(512,)
            reward = self.env.reward_history[i]
            action = self.env.action_history[i]

            #print('obs', np.array(obs).shape)
            #self.replay_buffer.add_sample({
            #    'obs': obs,
            #    'actions': [action],
            #    'rewards': [reward],
            #    'new_obs': i == len(self.env.reward_history)
            #    })

            if reward > 0 and action[6]:
                self.history.append((embedding, reward, action))

        save_interval = 2
        if self.recorded_episode_count % save_interval == 0:
            print('recording history')
            #pickle.dump(self.history, open('./history.pkl', 'wb'))
            self.annoy_index = AnnoyIndex(512)
            for i in range(len(self.history)):
                self.annoy_index.add_item(i, self.history[i][0])
            self.annoy_index.build(20)
            #pickle.dump(self.replay_buffer, open( "./replay_buffer.p", "wb" ))



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


        trim_length = random.randint(0, sequence_length // 5)
        del mutated[-trim_length:]
        print('trimmed %d of %d actions' % (trim_length, sequence_length))
        
        mutation_start_index = len(mutated)
        for i, action in reversed(list(enumerate(sequence[0:mutation_start_index]))):
            #percent_distance = i + 1 / sequence_length
            exponent = -(mutation_start_index - i - 1) / 1e2
            if random.random() < np.exp(exponent) * mutation_rate:
                #mutated = mutated[0:i]
                #print('trimmed %d of %d actions' % (mutation_start_index - len(mutated), sequence_length))
                #return mutated

                mutated[i] = random.choice(ACTIONS).copy()
                mutation_count += 1
        print('mutated %d out of %d actions' % (mutation_count, sequence_length))

        return mutated

    def random_next_step(self, left=False, jump_prob=1.0 / 10.0, jump_repeat=4, jumping_steps_left=0):
        action = random.choice(ACTIONS).copy()  #np.zeros((12,), dtype=np.bool)
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
        #random_prob = 0.5
        use_memory = random.random() > 0.5
        use_model = random.random() > 0.5
        times = {}
        while not done and steps_taken < num_steps:
            if self.model \
                    and self.annoy_index is not None \
                    and len(self.env.obs_history) > 0 \
                    and not left \
                    and use_memory \
                    and self.recorded_episode_count > 5:
                #print('sample', time.clock() - start_time)
                ob = [self.env.obs_history[-1]]

                embedding = self.session.run([self.output_layer], {self.model.X: ob})[0].reshape(512,)
                results = self.annoy_index.get_nns_by_vector(embedding, 100, include_distances=True)
                items = [self.history[i] for i in results[0]]
                rewards = [item[1] for item in items]
                #action = self.history[results[0][np.argmax(np.multiply(rewards, np.divide(1, results[1] + 1e9)))]][2]

                if len(rewards) > 0:
                    action = self.history[results[0][np.argmax(rewards)]][2]
                else:
                    action, jumping_steps_left = self.random_next_step(left, jump_prob, jump_repeat, jumping_steps_left)
                #print(action, 'memory')
                _, rew, done, _ = env.step(action)

                #print('step', time.clock() - start_time)

            elif self.model \
                    and len(self.env.obs_history) > 0 \
                    and not left \
                    and use_model:

                ob = [self.env.obs_history[-1]]
                actions = self.session.run([self.a0], {self.model.X: ob})
                action = ACTIONS[actions[0][0]].copy()
                #print(action, 'model')
                _, rew, done, _ = env.step(action)

            else:
                action, jumping_steps_left = self.random_next_step(left, jump_prob, jump_repeat, jumping_steps_left)
                #print(action, 'random')
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

def launch_env(game, state):
    #game, state = random.choice(env_data)
    # retro-contest-remote run -s tmp/sock -m monitor -d SonicTheHedgehog-Genesis GreenHillZone.Act1
    base_dir = './remotes/'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(base_dir + state, exist_ok=True)
    socket_dir = base_dir + "{}/sock".format(state)
    os.makedirs(socket_dir, exist_ok=True)
    monitor_dir = base_dir + "{}/monitor".format(state)
    os.makedirs(monitor_dir, exist_ok=True)
    subprocess.Popen(["retro-contest-remote", "run", "-s", socket_dir, '-m', monitor_dir, '-d', game, state], stdout=subprocess.PIPE)
    return grc.RemoteEnv(socket_dir)

def main():
    """Run JERK on the attached environment."""
    #env = grc.RemoteEnv('tmp/sock')
    env = launch_env('SonicTheHedgehog-Genesis', 'SpringYardZone.Act1')
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
