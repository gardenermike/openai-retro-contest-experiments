#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import DistQNetwork, NatureDistQNetwork
from anyrl.models.dqn_scalar import noisy_net_dense
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from sonic_util_train import AllowBacktracking, SonicDiscretizer, RewardScaler, FrameStack, WarpFrame, make_env
import csv
import retro
import random
import time
import pickle
import gzip
from functools import partial
import math

from keras.layers import Input, Dense, TimeDistributed, Lambda, Softmax
from keras.layers.merge import Multiply
from keras.layers.core import *
from keras.layers.convolutional import *
import joblib

def make_training_env(game, state, stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = retro.make(game=game, state=state)
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)

    return env

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


def make_training_envs():
    envs = []
    with open('./sonic-train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            game, state = row
            if game == 'game':
                continue
            else:
                env = make_training_env(game, state, stack=False, scale_rew=False)
                env = AllowBacktracking(env)
                envs.append([env])

def prep_env(env):
    env = AllowBacktracking(env)
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)

    return env

def train(dqn,
          num_steps,
          player,
          replay_buffer,
          optimize_op,
          train_interval=1,
          target_interval=8192,
          batch_size=32,
          min_buffer_size=20000,
          tf_schedules=(),
          handle_ep=lambda steps, rew: None,
          handle_step=lambda steps, rew: None,
          timeout=None,
          summary_op=None):
    """
    Run an automated training loop.
    This is meant to provide a convenient way to run a
    standard training loop without any modifications.
    You may get more flexibility by writing your own
    training loop.
    Args:
      num_steps: the number of timesteps to run.
      player: the Player for gathering experience.
      replay_buffer: the ReplayBuffer for experience.
      optimize_op: a TF Op to optimize the model.
      train_interval: timesteps per training step.
      target_interval: number of timesteps between
        target network updates.
      batch_size: the size of experience mini-batches.
      min_buffer_size: minimum replay buffer size
        before training is performed.
      tf_schedules: a sequence of TFSchedules that are
        updated with the number of steps taken.
      handle_ep: called with information about every
        completed episode.
      timeout: if set, this is a number of seconds
        after which the training loop should exit.
    """
    sess = dqn.online_net.session
    sess.run(dqn.update_target)
    steps_taken = 0
    total_reward = 0
    next_target_update = target_interval
    next_train_step = train_interval
    start_time = time.time()
    summary = None
    while steps_taken < num_steps:
        #if steps_taken % 1000 == 0:
        #    print('steps taken: {}'.format(steps_taken))
        #    print('total_reward: {}'.format(total_reward))
        if timeout is not None and time.time() - start_time > timeout:
            return
        transitions = player.play()
        for trans in transitions:
            total_reward += trans['total_reward']
            #handle_step(trans['episode_step'] + 1, trans['total_reward'])
            if trans['is_last'] or (steps_taken % 1000 == 0):
                total_reward = trans['total_reward']
                print('total_reward', total_reward) if total_reward > 0. else None
                #pass
                #print('steps: ', trans['episode_step'] + 1)
                #print('reward: ', trans['total_reward'])
                #handle_ep(trans['episode_step'] + 1, trans['total_reward'])
                #handle_step(trans['episode_step'] + 1, trans['total_reward'])
            #print(trans)
            replay_buffer.add_sample(trans)
            steps_taken += 1
            for sched in tf_schedules:
                sched.add_time(sess, 1)
            if replay_buffer.size >= min_buffer_size and steps_taken >= next_train_step:
                next_train_step = steps_taken + train_interval
                batch = replay_buffer.sample(batch_size)
                _, losses, summary = sess.run((optimize_op, dqn.losses, summary_op),
                                     feed_dict=dqn.feed_dict(batch))
                replay_buffer.update_weights(batch, losses)
            if steps_taken >= next_target_update:
                next_target_update = steps_taken + target_interval
                sess.run(dqn.update_target)

    return summary

def product(vals):
    """
    Compute the product of values in a list-like object.
    """
    prod = 1
    for val in vals:
        prod *= val
    return prod

# the default rainbow model
def nature_cnn(obs_batch, dense=tf.layers.dense):
    """
    Apply the CNN architecture from the Nature DQN paper.
    The result is a batch of feature vectors.
    """
    print(obs_batch)
    conv_kwargs = {
        'activation': tf.nn.relu,
        'kernel_initializer': tf.orthogonal_initializer(gain=math.sqrt(2))
    }
    with tf.variable_scope('layer_1'):
        cnn_1 = tf.layers.conv2d(obs_batch, 32, 8, 4, **conv_kwargs)
    with tf.variable_scope('layer_2'):
        cnn_2 = tf.layers.conv2d(cnn_1, 64, 4, 2, **conv_kwargs)
    with tf.variable_scope('layer_3'):
        cnn_3 = tf.layers.conv2d(cnn_2, 64, 3, 1, **conv_kwargs)
    flat_size = product([x.value for x in cnn_3.get_shape()[1:]])
    flat_in = tf.reshape(cnn_3, (tf.shape(cnn_3)[0], int(flat_size)))
    return dense(flat_in, 512, **conv_kwargs)
    # <tf.Variable 'online/layer_1/conv2d/kernel:0' shape=(8, 8, 16, 32) dtype=float32_ref>, <tf.Variable 'online/layer_1/conv2d/bias:0' shape=(32,) dtype=float32_ref>, <tf.Variable 'online/layer_2/conv2d/kernel:0' shape=(4, 4, 32, 64) dtype=float32_ref>, <tf.Variable 'online/layer_2/conv2d/bias:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'online/layer_3/conv2d/kernel:0' shape=(3, 3, 64, 64) dtype=float32_ref>, <tf.Variable 'online/layer_3/conv2d/bias:0' shape=(64,) dtype=float32_ref>

class CustomDistQNetwork(NatureDistQNetwork):

    def WaveNet_activation(self, x):
        tanh_out = Activation('tanh')(x)
        sigm_out = Activation('sigmoid')(x)  
        return Multiply()([tanh_out, sigm_out])

    def ED_TCN(self, obs_batch, dense=tf.layers.dense):
        filter_counts = [32, 64, 64]
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]
        n_layers = len(filter_counts)

        x = obs_batch
        #print(x)

        # Encoder
        x = Conv2D(filter_counts[0], kernel_sizes[0], padding='same')(x)
        x = SpatialDropout2D(0.1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(strides[0])(x)
        #print(x)

        x = Conv2D(filter_counts[1], kernel_sizes[1], padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(strides[1])(x)
        #print(x)

        x = Conv2D(filter_counts[2], kernel_sizes[2], padding='same')(x)
        x = self.WaveNet_activation(x)
        #print(x)

        #attention
        x = TimeDistributed(Dense(4))(x)
        x = Softmax(axis=-1)(x)
        #print(x)

        #for i in range(n_layers):
        #    x = Conv2D(filter_counts[i], kernel_sizes[i], padding='same')(x)
        #    x = SpatialDropout2D(0.3)(x)

        #    x = self.WaveNet_activation(x)

        #    if strides[i] > 1:
        #        x = MaxPooling2D(strides[i])(x)

        # Decoder
        #for i in range(n_layers):
        #    x = UpSampling2D(2)(x)
        #    x = Conv2D(n_nodes[-i-1], kernel_size - (n_layers - i - 1) * 2, padding='same')(x)
        #    x = self.WaveNet_activation(x)

        #x = TimeDistributed(Dense(16, activation="softmax" ))(x)


        flat_size = product([dim.value for dim in x.get_shape()[1:]])
        flat_in = tf.reshape(x, (tf.shape(x)[0], int(flat_size)))

        conv_kwargs = {
            'activation': tf.nn.relu,
            'kernel_initializer': tf.orthogonal_initializer(gain=math.sqrt(2))
        }
        return dense(flat_in, 256, **conv_kwargs)

    def stacked_cnn(self, obs_batch):
        x = obs_batch

        initializer = tf.orthogonal_initializer(gain=math.sqrt(2))
        # activation=WaveNet_activation
        activation = 'relu'
        y = Conv2D(32, kernel_size=8, strides=4, activation=activation, kernel_initializer=initializer, name='layer_1')(x)
        y = Dropout(0.2)(y)
        y = Conv2D(64, kernel_size=4, strides=2, activation=activation, kernel_initializer=initializer, name='layer_2')(y)
        y = Dropout(0.1)(y)
        y = Conv2D(64, kernel_size=3, strides=1, activation=activation, kernel_initializer=initializer, name='layer_3')(y)
        y = Dropout(0.1)(y)

        y = Flatten(name='flatten')(y)
        y = Dense(512, activation='relu', kernel_initializer=initializer, name='dense1')(y)

        return y

    def preprocess_batch(self, obs_batch, stack=False):
        obs_batch = tf.cast(obs_batch, tf.float32)
        obs_batch = obs_batch / 255.
        obs_batch -= 0.5

        if False: #stack:
            obs_batch = tf.concat(obs_batch, axis=-1)

        return obs_batch

    def base(self, obs_batch):
        obs_batch = self.preprocess_batch(obs_batch, stack=False)
        #obs_batch = tf.cast(obs_batch, tf.float32) * self.input_scale
        
        return nature_cnn(obs_batch, dense=self.dense)
        #return self.ED_TCN(obs_batch)
        #return self.stacked_cnn(obs_batch)

def models(session,
                   num_actions,
                   obs_vectorizer,
                   num_atoms=51,
                   min_val=-10,
                   max_val=10,
                   sigma0=0.5):
    """
    Tinker with the models used for Rainbow
    (https://arxiv.org/abs/1710.02298).
    Args:
      session: the TF session.
      num_actions: size of action space.
      obs_vectorizer: observation vectorizer.
      num_atoms: number of distribution atoms.
      min_val: minimum atom value.
      max_val: maximum atom value.
      sigma0: initial Noisy Net noise.
    Returns:
      A tuple (online, target).
    """
    maker = lambda name: CustomDistQNetwork(session, num_actions, obs_vectorizer, name,
                                            num_atoms, min_val, max_val, dueling=True,
                                            dense=partial(noisy_net_dense, sigma0=sigma0))
    return maker('online'), maker('target')



def main():
    """Run DQN until the environment throws an exception."""
    #env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    #envs = make_training_envs()
    #env = BatchedFrameStack(BatchedGymEnv(envs), num_images=4, concat=False)
    #env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    
    envs = get_training_envs()
    game, state = random.choice(envs)
    env = make_training_env(game, state, stack=False, scale_rew=False)
    env = prep_env(env)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        dqn = DQN(*models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        loss = dqn.loss
        train_writer = tf.summary.FileWriter( './logs/multiple/train', sess.graph)
        tf.summary.scalar("loss", loss)
        reward = tf.Variable(0., name='reward', trainable=False)
        tf.summary.scalar('reward', tf.reduce_mean(reward))
        steps = tf.Variable(0, name='steps', trainable=False)
        tf.summary.scalar('steps', tf.reduce_mean(steps))
        summary_op = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        print(tf.trainable_variables())

        #graph = tf.get_default_graph()
        #restore_saver = tf.train.Saver({
        #    'dense1/bias': graph.get_tensor_by_name('online/dense1/bias:0'),
        #    'dense1/kernel': graph.get_tensor_by_name('online/dense1/kernel:0'),
        #    'layer_1/bias': graph.get_tensor_by_name('online/layer_1/bias:0'),
        #    'layer_1/kernel': graph.get_tensor_by_name('online/layer_1/kernel:0'),
        #    'layer_2/bias': graph.get_tensor_by_name('online/layer_2/bias:0'),
        #    'layer_2/kernel': graph.get_tensor_by_name('online/layer_2/kernel:0'),
        #    'layer_3/bias': graph.get_tensor_by_name('online/layer_3/bias:0'),
        #    'layer_3/kernel': graph.get_tensor_by_name('online/layer_3/kernel:0'),
        #    'dense1/bias': graph.get_tensor_by_name('online_1/dense1/bias:0'),
        #    'dense1/kernel': graph.get_tensor_by_name('online_1/dense1/kernel:0'),
        #    'layer_1/bias': graph.get_tensor_by_name('online_1/layer_1/bias:0'),
        #    'layer_1/kernel': graph.get_tensor_by_name('online_1/layer_1/kernel:0'),
        #    'layer_2/bias': graph.get_tensor_by_name('online_1/layer_2/bias:0'),
        #    'layer_2/kernel': graph.get_tensor_by_name('online_1/layer_2/kernel:0'),
        #    'layer_3/bias': graph.get_tensor_by_name('online_1/layer_3/bias:0'),
        #    'layer_3/kernel': graph.get_tensor_by_name('online_1/layer_3/kernel:0'),
        #    'dense1/bias': graph.get_tensor_by_name('online_2/dense1/bias:0'),
        #    'dense1/kernel': graph.get_tensor_by_name('online_2/dense1/kernel:0'),
        #    'layer_1/bias': graph.get_tensor_by_name('online_2/layer_1/bias:0'),
        #    'layer_1/kernel': graph.get_tensor_by_name('online_2/layer_1/kernel:0'),
        #    'layer_2/bias': graph.get_tensor_by_name('online_2/layer_2/bias:0'),
        #    'layer_2/kernel': graph.get_tensor_by_name('online_2/layer_2/kernel:0'),
        #    'layer_3/bias': graph.get_tensor_by_name('online_2/layer_3/bias:0'),
        #    'layer_3/kernel': graph.get_tensor_by_name('online_2/layer_3/kernel:0'),
        #    'dense1/bias': graph.get_tensor_by_name('target/dense1/bias:0'),
        #    'dense1/kernel': graph.get_tensor_by_name('target/dense1/kernel:0'),
        #    'layer_1/bias': graph.get_tensor_by_name('target/layer_1/bias:0'),
        #    'layer_1/kernel': graph.get_tensor_by_name('target/layer_1/kernel:0'),
        #    'layer_2/bias': graph.get_tensor_by_name('target/layer_2/bias:0'),
        #    'layer_2/kernel': graph.get_tensor_by_name('target/layer_2/kernel:0'),
        #    'layer_3/bias': graph.get_tensor_by_name('target/layer_3/bias:0'),
        #    'layer_3/kernel': graph.get_tensor_by_name('target/layer_3/kernel:0'),
        #    })
        #restore_saver.restore(sess, './model-images/model.ckpt')
        #print('model restored')

        weights = joblib.load('./ppo2_weights_266.joblib')
        #[<tf.Variable 'model/c1/w:0' shape=(8, 8, 4, 32) dtype=float32_ref>, <tf.Variable 'model/c1/b:0' shape=(1, 32, 1, 1) dtype=float32_ref>, <tf.Variable 'model/c2/w:0' shape=(4, 4, 32, 64) dtype=float32_ref>, <tf.Variable 'model/c2/b:0' shape=(1, 64, 1, 1) dtype=float32_ref>, <tf.Variable 'model/c3/w:0' shape=(3, 3, 64, 64) dtype=float32_ref>, <tf.Variable 'model/c3/b:0' shape=(1, 64, 1, 1) dtype=float32_ref>, <tf.Variable 'model/fc1/w:0' shape=(3136, 512) dtype=float32_ref>, <tf.Variable 'model/fc1/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'model/v/w:0' shape=(512, 1) dtype=float32_ref>, <tf.Variable 'model/v/b:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'model/pi/w:0' shape=(512, 7) dtype=float32_ref>, <tf.Variable 'model/pi/b:0' shape=(7,) dtype=float32_ref>]

        graph = tf.get_default_graph()
        for model in ['online', 'target']:
            tensor_names = [
                    '{}/layer_1/conv2d/kernel:0',
                    '{}/layer_1/conv2d/bias:0',
                    '{}/layer_2/conv2d/kernel:0',
                    '{}/layer_2/conv2d/bias:0',
                    '{}/layer_3/conv2d/kernel:0',
                    '{}/layer_3/conv2d/bias:0',
                    #'{}/dense1/kernel:0',
                    #'{}/dense1/bias:0'
                    ]
            for i in range(len(tensor_names)):
                tensor_name = tensor_names[i].format(model)
                tensor = graph.get_tensor_by_name(tensor_name)
                weight = weights[i]
                if 'bias' in tensor_name:
                    weight = np.reshape(weight, tensor.get_shape())
                print('about to assign {} value with size {}'.format(tensor_name, weights[i].shape))
                sess.run(tf.assign(tensor, weight))
            

        saver = tf.train.Saver()
        save_path = saver.save(sess, "./model/model.ckpt")
        print('Saved model')
        replay_buffer = PrioritizedReplayBuffer(100000, 0.5, 0.4, epsilon=0.1)


        #replay_buffer = pickle.load(gzip.open('./docker-build/model/replay_buffer.p.gz', 'rb'))
        #replay_buffer = pickle.load(open('./model/replay_buffer.p', 'rb'))

        total_steps = 50000000
        steps_per_env = 5000
        env.close()

        for i in range(int(total_steps / steps_per_env)):
            game, state = random.choice(envs)
            env = make_training_env(game, state, stack=False, scale_rew=False)
            env = prep_env(env)
            player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)

            #dqn.train(num_steps=steps_per_env, # Make sure an exception arrives before we stop.
            #      player=player,
            #      replay_buffer=replay_buffer,
            #      optimize_op=optimize,
            #      train_interval=1,
            #      target_interval=8192,
            #      batch_size=32,
            #      min_buffer_size=20000)
            
            summary = train(dqn,
                  num_steps=steps_per_env, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=replay_buffer,
                  optimize_op=optimize,
                  train_interval=4,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000,
                  summary_op=summary_op,
                  handle_ep=lambda st, rew: (reward.assign(rew), steps.assign(st)),
                  handle_step=lambda st, rew: (reward.assign(reward + rew), steps.assign(steps + st)))


            env.close()

            if summary:
                train_writer.add_summary(summary, i)
            else:
                print('No summary')

            save_path = saver.save(sess, "./model/model.ckpt")
            pickle.dump(replay_buffer, open( "./model/replay_buffer.p", "wb" ))
            print('Saved model')

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
