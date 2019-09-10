import tensorflow as tf
from abc import abstractmethod
import time
import os


class Optimizer(object):

    def __init__(self, model, train_set, evaluator, validation_set=None, **kwargs):
        self.model = model
        self.train_set = train_set
        self.validation_set = validation_set
        self.evaluator = evaluator

        self.input_shape = model.net['input'].get_shape()[1]
        self.batch_size = kwargs.pop('batch_size', 256)
        self.num_epochs = kwargs.pop('num_epochs', 1000)
        self.initial_learning_rate = kwargs.pop('initial_learning_rate', 0.01)

        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        self.optimize_op = self._optimize_op()

        self.curr_epoch = 0
        self.num_idiot_epochs = 0
        self.best_score = evaluator.worst_score
        self.curr_learning_rate = self.initial_learning_rate

    @abstractmethod
    def _optimize_op(self, **kwargs):
        pass

    @abstractmethod
    def _update_learning_rate(self, **kwargs):
        pass

    def _step(self, sess, **kwargs):
        augment = kwargs.pop('augment_train', True)

        batch_xs, batch_ys = self.train_set.next_batch(self.batch_size, input_shape=self.input_shape,
                                                       shuffle=True, augment=augment, is_training=True)
        y_true = batch_ys
        _, loss, y_pred, y_logit = sess.run([self.optimize_op, self.model.loss, self.model.pred, self.model.logits],
                                            feed_dict={self.model.X: batch_xs,
                                                       self.model.Y: batch_ys,
                                                       self.model.is_training: True,
                                                       self.learning_rate: self.curr_learning_rate})
        self.curr_epoch = self.train_set.epochs_completed
        return loss, y_true, y_pred

    def train(self, sess, verbose=False, **kwargs):
        save_dir = kwargs.pop('save_dir', './checkpoints')

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        step_losses = []
        step_train_scores = []
        step_validation_scores = []
        total_training_time = 0
        epochs_completed = 0
        while True:
            tick = time.time()
            step_loss, step_y_true, step_y_pred = self._step(sess, **kwargs)
            step_losses.append(step_loss)
            step_train_time = time.time() - tick
            total_training_time += step_train_time
            step_train_score = self.evaluator.score(step_y_true, step_y_pred)
            step_train_scores.append(step_train_score)

            step = sess.run(self.global_step)
            num_images = step * self.batch_size

            if self.curr_epoch > epochs_completed:
                if self.validation_set is not None:
                    validation_y_true = self.validation_set.labels
                    validation_y_pred = self.model.predict(sess, self.validation_set, **kwargs)
                    validation_score = self.evaluator.score(validation_y_true, validation_y_pred)
                    step_validation_scores.append(validation_score)
                    curr_score = validation_score
                    print('{} epoch, {} step, {:.6f}, {:.6f} avg, {:.4f} avg, {:.6f} rate, {:.6f} secs, {} images' \
                          .format(self.curr_epoch, step, step_loss, step_train_score, validation_score,
                                  self.curr_learning_rate, step_train_time, num_images))
                else:
                    curr_score = step_train_score
                    print('{} epoch, {} step, {:.6f}, {:.6f} avg, {:.6f} rate, {:.6f} secs, {} images' \
                          .format(self.curr_epoch, step, step_loss, step_train_score,
                                  self.curr_learning_rate, step_train_time, num_images))

                if self.evaluator.is_better(curr=curr_score, best=self.best_score, **kwargs):
                    self.best_score = curr_score
                    self.num_idiot_epochs = 0
                    saver.save(sess, os.path.join(save_dir, 'model'), global_step=self.global_step)
                    if verbose:
                        print('The best model was updated and saved. {:.6f} avg.'.format(self.best_score))
                else:
                    self.num_idiot_epochs += 1

                if not self.curr_epoch < self.num_epochs:
                    break
                self._update_learning_rate(**kwargs)

            epochs_completed = self.curr_epoch

        if verbose:
            print('Total training time(s): {:.3f}'.format(total_training_time))
            print('Total epochs/steps: {}/{}'.format(epochs_completed, step))
            print('Best {} score: {:.3f}'.format('evaluation' if self.validation_set is not None else 'training',
                                                 self.best_score))
            print('Done.')

        rst = dict()
        rst['step_losses'] = step_losses
        rst['step_train_scores'] = step_train_scores
        if self.validation_set is not None:
            rst['validation_scores'] = step_validation_scores

        return rst


class MomentumOptimizer(Optimizer):

    def _optimize_op(self, **kwargs):
        momentum = kwargs.pop('momentum', 0.9)
        var_list = tf.trainable_variables()
        return tf.train.MomentumOptimizer(self.learning_rate, momentum=momentum, use_nesterov=False) \
            .minimize(loss=self.model.loss, var_list=var_list)

    def _update_learning_rate(self, **kwargs):
        learning_rate_patience = kwargs.pop('learning_rate_patience', 10)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.1)
        eps = kwargs.pop('learning_rate_eps', 1e-8)

        if self.num_idiot_epochs > learning_rate_patience:
            new_learning_rate = self.curr_learning_rate * learning_rate_decay
            if self.curr_learning_rate - new_learning_rate > eps:
                self.curr_learning_rate = new_learning_rate
            self.num_idiot_epochs = 0


class AdamOptimizer(Optimizer):

    def _optimize_op(self, **kwargs):
        momentum = kwargs.pop('momentum', 0.9)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        var_list = tf.trainable_variables()
        with tf.control_dependencies(extra_update_ops):
            train_op = tf.train.AdamOptimizer(self.learning_rate, momentum).minimize(
                self.model.loss, global_step=self.global_step, var_list=var_list)
        return train_op

    def _update_learning_rate(self, **kwargs):
        # TODO: polynomial rate decay implementation.
        learning_rate_patience = kwargs.pop('learning_rate_patience', 10)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.1)
        eps = kwargs.pop('learning_rate_eps', 1e-8)

        if self.num_idiot_epochs > learning_rate_patience:
            new_learning_rate = self.curr_learning_rate * learning_rate_decay
            if self.curr_learning_rate - new_learning_rate > eps:
                self.curr_learning_rate = new_learning_rate
            self.num_idiot_epochs = 0
