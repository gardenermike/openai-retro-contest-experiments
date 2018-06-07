#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from sonic_util import AllowBacktracking, make_env
import pickle
import gzip
import joblib
import numpy as np

def restore_ppo2_weights(sess):
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
                ]
        for i in range(len(tensor_names)):
            tensor_name = tensor_names[i].format(model)
            tensor = graph.get_tensor_by_name(tensor_name)
            weight = weights[i]
            if 'bias' in tensor_name:
                weight = np.reshape(weight, tensor.get_shape())
            #print('about to assign {} value with size {}'.format(tensor_name, weights[i].shape))
            sess.run(tf.assign(tensor, weight))

def main():
    """Run DQN until the environment throws an exception."""
    env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, "/root/compo/model.ckpt")
        #print('model restored')
        replay_buffer = pickle.load(gzip.open('/root/compo/replay_buffer.p.gz', 'rb'))
        replay_buffer.alpha = 0.2
        replay_buffer.beta = 0.4
        replay_buffer.capacity = 100000

        restore_ppo2_weights(sess)

        dqn.train(num_steps=2000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=replay_buffer, #PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=4,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
