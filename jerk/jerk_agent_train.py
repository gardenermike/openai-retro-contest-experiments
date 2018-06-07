#!/usr/bin/env python

"""
A scripted agent called "Just Enough Retained Knowledge".
"""

import random
import copy
import sys
import os
import subprocess
import threading

import gym
import numpy as np

import gym_remote.client as grc
import gym_remote.exceptions as gre

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

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

def get_training_envs():
    envs = []
    with open('./sonic-train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            game, state = row
            if game == 'game':
                continue
            else:
                envs.append(row)
    return envs

class DelegatingDummyVecEnv(DummyVecEnv):
    def __getattr__(self, attr):
        return getattr(self.envs[0], attr)


class JerkAgent(threading.Thread):
    def __init__(self, env, solutions):
        threading.Thread.__init__(self)
        self.env = env
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
        env = TrackedEnv(self.env)
        new_ep = True
        best_reward = 0.
        best_pair = None

        while True:
            if new_ep:
                if len(self.solutions) > 0:
                    best_pair, best_reward = self.best_solution()

                if self.solutions and self.should_use_history(best_reward, env):
                    #solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                    #best_pair = solutions[-1]
                    new_rew, last_reward_index = self.exploit(env, best_pair[1])
                    best_pair[0].append(new_rew)
                    if (new_rew > best_reward) and len(env.best_sequence()) != len(best_pair[1]):
                        self.solutions.append(([max(env.reward_history)], env.best_sequence()[0:last_reward_index]))
                    print('replayed best with reward %f' % new_rew)
                    continue
                elif best_pair:
                    mutation_rate = (1 - (best_reward / MAX_SCORE)) * MUTATION_DAMPEN
                    mutated = self.mutate(best_pair[1], mutation_rate)
                    new_rew, last_reward_index = self.exploit(env, mutated)
                    print('mutated solution rewarded %f vs %f' % (new_rew, best_reward))
                    if new_rew > best_reward:
                        self.solutions.append(([max(env.reward_history)], env.best_sequence()[0:last_reward_index]))
                    continue
                else:
                    env.reset()
                    new_ep = False
            rew, new_ep = self.move(env, 100)
            if not new_ep and rew <= 0:
                print('backtracking due to negative reward: %f' % rew)
                _, new_ep = self.move(env, 70, left=True)
            if new_ep:
                self.solutions.append(([max(env.reward_history)], env.best_sequence()))

    def mutate(self, sequence, mutation_rate):
        mutated = copy.deepcopy(sequence)
        sequence_length = len(sequence)
        mutation_count = 0

        mutation_start_index = min(sequence_length, random.randint(100, 2000))

        for i, action in reversed(list(enumerate(sequence[0:mutation_start_index]))):
            #percent_distance = i + 1 / sequence_length
            exponent = -(mutation_start_index - i - 1) / 1e2
            if random.random() < np.exp(exponent) * mutation_rate:
                mutated = mutated[0:i]
                print('trimmed %d of %d actions' % (sequence_length - i, sequence_length))
                return mutated

                mutated[i] = random.choice(ACTIONS)
                mutation_count += 1
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
        total_rew = 0.0
        done = False
        steps_taken = 0
        jumping_steps_left = 0
        while not done and steps_taken < num_steps:
            action, jumping_steps_left = self.random_next_step(left, jump_prob, jump_repeat, jumping_steps_left)
            _, rew, done, _ = env.step(action)
            total_rew += rew
            steps_taken += 1
            if done:
                break
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


class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        self.action_history = []
        self.reward_history = []
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
        self.total_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        return obs, rew, done, info


def main():
    #env = grc.RemoteEnv('tmp/sock')
    #env_data = get_training_envs()
    if len(sys.argv) < 4:
        print('Usage: python jerk_agent_train.py game state process_count')
        sys.exit()

    game = sys.argv[1]
    state = sys.argv[2]
    process_count = sys.argv[3]

    sockets = []
    for index in range(int(process_count)):
        #game, state = random.choice(env_data)
        # retro-contest-remote run -s tmp/sock -m monitor -d SonicTheHedgehog-Genesis GreenHillZone.Act1
        state_directory_name = state + '-' + str(index)
        base_dir = './remotes/'
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(base_dir + state_directory_name, exist_ok=True)
        socket_dir = base_dir + "{}/sock".format(state_directory_name)
        os.makedirs(socket_dir, exist_ok=True)
        monitor_dir = base_dir + "{}/monitor".format(state_directory_name)
        os.makedirs(monitor_dir, exist_ok=True)
        subprocess.Popen(["retro-contest-remote", "run", "-s", socket_dir, '-m', monitor_dir, '-d', game, state], stdout=subprocess.PIPE)
        print('launched {} ({})'.format(state, index))

        sockets.append(socket_dir)
        #envs.append(lambda: make_env(socket_dir=state + '/sock'))
    print('remote processes launched')
    #env = lambda: make_training_env('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1', stack=True, scale_rew=True)
    #env = MultigameEnvWrapper
    #load_path = '/root/compo/trained_on_images_nature_cnn.joblib'
    #load_path = './saved_weights.joblib'
    #logger.configure(dir='./logs', format_strs=['stdout', 'tensorboard'])

    print('training...')

    global_solutions = []

    agents = []
    for socket_dir in sockets:
        env = grc.RemoteEnv(socket_dir)
        agent = JerkAgent(env, global_solutions)
        agents.append(agent)

    for agent in agents:
        agent.start()

    print('Created {} agents'.format(len(agents)))

    for agent in agents:
        agent.join()

    #def launch_env(socket_dir):
    #    return lambda: grc.RemoteEnv(socket_dir)

    #env = DelegatingDummyVecEnv([launch_env(socket_dir) for socket_dir in sockets])

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
