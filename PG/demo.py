"""
ppo alg
"""

import math
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from keras import backend as K
from keras.regularizers import l2

from keras.models import load_model, Model
from keras.layers import Dense
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import keras.backend.tensorflow_backend as K


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


class PPO:
    """
    portable ppo
    """

    def __init__(self, n_action, dim_ob_image,
                 pretrain_model='/data/mahjong/models/base_large2/drop.h5'):
        self.n_action = n_action
        self.dim_ob_image = dim_ob_image
        self.pretrain_model = pretrain_model
        self.discount = 0.99
        self.tau = 0.95
        self.entropy_coefficient = 0.01
        self.critic_coefficient = 1.0
        self.max_grad_norm = 0.5
        self.training_epoch = 10
        self.training_batchsize = 64


        self.lr_schedule = lambda x: (1 - x) * 2.5e-4
        self.clip_schedule = lambda x: (1 - x) * 0.1

        self.summary_writter = None
        tf.reset_default_graph()
        tf.Variable(0, name="global_step", trainable=False)
        self.saver = tf.train.Saver(max_to_keep=20)
        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True  # pylint: disable=E1101
        self.sess = tf.Session(config=conf)
        K.set_session(self.sess)
        self._build_network()
        self._build_algorithm()
        self._prepare()

    def _prepare(self):

        def initialize_uninitialized(sess):
            global_vars = tf.global_variables()
            is_not_initialized = sess.run(
                [tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(
                global_vars, is_not_initialized) if not f]
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))

        initialize_uninitialized(self.sess)

    def save_model(self, filename):
        """Save model to `save_path`."""
        global_step = self.sess.run(tf.train.get_global_step())
        self.saver.save(
            self.sess,
            filename,
            global_step,
            write_meta_graph=True
        )

    def load_model(self, filename):
        """Load model from `save_path` if there exists."""
        latest_checkpoint = tf.train.latest_checkpoint(filename)
        if latest_checkpoint:
            self.saver.restore(self.sess, latest_checkpoint)
            return True
        else:
            return False

    def _build_network(self):
        self.ob_image = tf.placeholder(
            tf.uint8, [None, *self.dim_ob_image], name="image_observation")
        # [None, n_action, actions_history] one-hot representation
        _cast_ob_image = tf.cast(self.ob_image, dtype=tf.float32)
        with tf.variable_scope("pretrain_model"):
            baseline_drop = load_model(self.pretrain_model)
        baseline_drop.layers.pop(0)
        baseline_drop.layers[-1].activation = None
        policy_output = baseline_drop(_cast_ob_image)

        tmp = baseline_drop.get_layer(baseline_drop.layers[-2].name).output
        value_dense_1 = Dense(100, activation='relu', name="value_dense1")(tmp)
        value_output = Dense(1, activation=None,
                             name="value_dense2")(value_dense_1)
        value_model = Model(baseline_drop.inputs[0], value_output)

        state_value = tf.squeeze(value_model(_cast_ob_image))

        self.logit_action_probability = policy_output
        self.state_value = state_value
        self.policy_model = baseline_drop
        self.value_model = value_model

    def _build_algorithm(self):
        self.init_clip_epsilon = 0.1
        self.init_lr = 2.5e-4
        self.clip_epsilon = tf.placeholder(tf.float32)
        self.moved_lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.moved_lr, epsilon=1e-5)

        self.old_logit_action_probability = tf.placeholder(
            tf.float32, [None, self.n_action])
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.target_state_value = tf.placeholder(
            tf.float32, [None], "target_state_value")

        # Get selected action index.
        batch_size = tf.shape(self.ob_image)[0]
        selected_action_index = tf.stack(
            [tf.range(batch_size), self.action], axis=1)

        # Compute entropy of the action probability.
        log_prob_1 = tf.nn.log_softmax(self.logit_action_probability)
        log_prob_2 = tf.stop_gradient(
            tf.nn.log_softmax(self.old_logit_action_probability))

        prob_1 = tf.nn.softmax(log_prob_1)
        prob_2 = tf.stop_gradient(tf.nn.softmax(log_prob_2))

        # entropy = - \sum_i p_i \log(p_i)
        self.entropy = - tf.reduce_sum(log_prob_1 * prob_1, axis=1)

        # Compute ratio of the action probability.
        logit_act1 = tf.gather_nd(log_prob_1, selected_action_index)
        logit_act2 = tf.gather_nd(log_prob_2, selected_action_index)

        self.ratio = tf.exp(logit_act1 - logit_act2)

        # Get surrogate object.
        surrogate_1 = self.ratio * self.advantage
        surrogate_2 = tf.clip_by_value(
            self.ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * self.advantage

        self.surrogate = -tf.reduce_mean(tf.minimum(surrogate_1, surrogate_2))

        # Compute critic loss.
        self.critic_loss = tf.reduce_mean(tf.squared_difference(
            self.state_value, self.target_state_value))

        # Compute gradients.
        self.total_loss = self.surrogate + self.critic_coefficient * \
            self.critic_loss - self.entropy_coefficient * self.entropy

        update_variables = tf.trainable_variables()
        update_variables = [i for i in update_variables if 'pretrain_model' not in i.name]

        grads = tf.gradients(self.total_loss, update_variables)

        # Clip gradients.
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.total_train_op = self.optimizer.apply_gradients(
            zip(clipped_grads, update_variables), global_step=tf.train.get_global_step())

    def inference(self, states, infos, temperature=1.0):

        n_inference = len(states)
        logit = self.sess.run(self.logit_action_probability,
                              feed_dict={self.ob_image: states})
        logit = logit - np.max(logit, axis=1, keepdims=True)
        prob = np.exp(logit / temperature) / \
            np.sum(np.exp(logit / temperature), axis=1, keepdims=True)
        for i in range(n_inference):
            for j in range(34):
                if infos[i]['valid_drop'][j] == 0:
                    prob[i, j] = 0
            prob[i] = prob[i] / np.sum(prob[i])
        action = [np.random.choice(self.n_action, p=prob[i, :])
                  for i in range(n_inference)]
        return action

    def update(self, s_batch, a_batch, r_batch, d_batch, update_ratio):

        advantage_batch, target_value_batch, old_logit_action_probability_batch = [], [], []
        for i in range(len(d_batch)):
            traj_size = len(d_batch[i])
            adv = np.empty(traj_size, dtype=np.float32)
            # print([type(j) for j in s_batch[i]])
            old_logit, state_value = self.sess.run(
                [self.logit_action_probability, self.state_value],
                feed_dict={self.ob_image: s_batch[i][:-1]})

            old_logit_action_probability_batch += old_logit.tolist()
            delta_value = r_batch[i] + self.discount * \
                (1 - d_batch[i]) * state_value - state_value

            last_advantage = 0

            for t in reversed(range(traj_size)):
                adv[t] = delta_value[t] + self.discount * \
                    self.tau * (1 - d_batch[i][t]) * last_advantage
                last_advantage = adv[t]

            # Compute target value.
            target_value_batch.append(state_value + adv)
            # Collect advantage.
            advantage_batch.append(adv)

        # Flat the batch values.
        advantage_batch = np.concatenate(advantage_batch, axis=0)
        target_value_batch = np.concatenate(target_value_batch, axis=0)
        all_step = sum(len(dones) for dones in d_batch)

        s_batch = np.concatenate([s[:-1] for s in s_batch], axis=0)
        a_batch = np.concatenate(a_batch, axis=0)
        advantage_batch = advantage_batch.reshape(all_step)
        target_value_batch = target_value_batch.reshape(all_step)

        # Normalize Advantage.
        advantage_batch = (advantage_batch - advantage_batch.mean()
                           ) / (advantage_batch.std() + 1e-5)

        old_logit_action_probability_batch = np.asarray(
            old_logit_action_probability_batch)

        # Train network.
        for _ in tqdm(range(self.training_epoch)):
            # Get training sample generator.
            batch_generator = self._generator(
                [s_batch, a_batch, advantage_batch, old_logit_action_probability_batch, target_value_batch], batch_size=self.training_batchsize)

            while True:
                try:
                    mini_s_batch, mini_a_batch, mini_advantage_batch, mini_old_logit_action_probability_batch, mini_target_state_value_batch = next(
                        batch_generator)

                    global_step = self.sess.run(tf.train.get_global_step())

                    fd = {
                        self.ob_image: mini_s_batch,
                        self.old_logit_action_probability: mini_old_logit_action_probability_batch,
                        self.action: mini_a_batch,
                        self.advantage: mini_advantage_batch,
                        self.target_state_value: mini_target_state_value_batch,
                        self.moved_lr: self.lr_schedule(update_ratio),
                        self.clip_epsilon: self.clip_schedule(update_ratio)}

                    c_loss, surr, entro, p_ratio, _ = self.sess.run([self.critic_loss,
                                                                     self.surrogate,
                                                                     self.entropy,
                                                                     self.ratio,
                                                                     self.total_train_op],
                                                                    feed_dict=fd)

                    # if global_step % 100 == 0:
                    #     print(
                    #         f"c_loss: {c_loss}  surr: {surr}  entro: {entro[0]}  ratio: {p_ratio[0]} at step {global_step}")

                except StopIteration:
                    del batch_generator
                    break

    def _generator(self, data_batch, batch_size=32):
        n_sample = data_batch[0].shape[0]
        index = np.arange(n_sample)
        np.random.shuffle(index)
        for i in range(math.ceil(n_sample / batch_size)):
            span_index = slice(
                i * batch_size, min((i + 1) * batch_size, n_sample))
            span_index = index[span_index]
            yield [x[span_index, :] if x.ndim > 1 else x[span_index] for x in data_batch]
