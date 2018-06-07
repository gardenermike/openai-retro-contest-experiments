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

def should_use_history(reward, env):
    reward_percentage = reward / MAX_SCORE

    the_end_is_nigh = (EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS) ** 3

    return random.random() < np.mean([reward_percentage, the_end_is_nigh])

def best_solution(solutions):
    best_pair = sorted(solutions, key=lambda x: np.mean(x[0]))[-1]
    reward = np.mean(best_pair[0])

    return best_pair, reward

def main():
    """Run JERK on the attached environment."""
    env = grc.RemoteEnv('tmp/sock')
    env = TrackedEnv(env)
    new_ep = True
    solutions = []
    keep_percentage = 0.6
    best_reward = 0.
    best_pair = None

    while True:
        if new_ep:
            if len(solutions) > 0:
                best_pair, best_reward = best_solution(solutions)
            #if (solutions and
            #        random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
            if solutions and should_use_history(best_reward, env):
                solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                best_pair = solutions[-1]
                new_rew, last_reward_index = exploit(env, best_pair[1])
                best_pair[0].append(new_rew)
                if (new_rew / best_reward > 0.6) and len(env.best_sequence()) != len(best_pair[1]):
                    solutions.append(([max(env.reward_history)], env.best_sequence()[0:last_reward_index]))
                print('replayed best with reward %f' % new_rew)
                continue
            elif best_pair:
                mutation_rate = (1 - (best_reward / MAX_SCORE)) * MUTATION_DAMPEN
                mutated = mutate(best_pair[1], mutation_rate)
                new_rew, last_reward_index = exploit(env, mutated)
                print('mutated solution rewarded %f vs %f' % (new_rew, best_reward))
                if (new_rew / best_reward > 0.6):
                    solutions.append(([max(env.reward_history)], env.best_sequence()[0:last_reward_index]))
                continue
            else:
                env.reset()
                new_ep = False
        rew, new_ep = move(env, 100)
        if not new_ep and rew <= 0:
            print('backtracking due to negative reward: %f' % rew)
            _, new_ep = move(env, 70, left=True)
        if new_ep:
            solutions.append(([max(env.reward_history)], env.best_sequence()))

def mutate(sequence, mutation_rate):
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

            mutated[i] = random.choice(ACTIONS)
            mutation_count += 1
    print('mutated %d out of %d actions' % (mutation_count, sequence_length))

    return mutated

def random_next_step(left=False, jump_prob=1.0 / 10.0, jump_repeat=4, jumping_steps_left=0):
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


def move(env, num_steps, left=False, jump_prob=1.0 / 10.0, jump_repeat=4):
    """
    Move right or left for a certain number of steps,
    jumping periodically.
    """
    total_rew = 0.0
    done = False
    steps_taken = 0
    jumping_steps_left = 0
    while not done and steps_taken < num_steps:
        action, jumping_steps_left = random_next_step(left, jump_prob, jump_repeat, jumping_steps_left)
        _, rew, done, _ = env.step(action)
        total_rew += rew
        steps_taken += 1
        if done:
            break
    return total_rew, done

def exploit(env, sequence):
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
                reward, done = move(env, steps, left)
                idx += steps
                if left:
                    left = False
                if reward == 0:
                    left = True
                else:
                    last_reward_index = idx
                #action, jumping_steps_left = random_next_step(jumping_steps_left=jumping_steps_left)
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

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
