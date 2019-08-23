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

        self.batch_size = kwargs.pop('batch_size', 256)
        self.num_epochs = kwargs.pop('num_epochs', 1000)
        self.initial_learning_rate = kwargs.pop('initial_learning_rate', 0.01)

        self.learning_rate = tf.placeholder(dtype=tf.float32)
        self.optimize_op = self._optimize_op()

        self.curr_epoch = 1
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

        batch_xs, batch_ys = self.train_set.next_batch(self.batch_size, shuffle=True, augment=augment, is_training=True)
        y_true = batch_ys
        _, loss, y_pred = sess.run([self.optimize_op, self.model.loss, self.model.pred],
                                   feed_dict={self.model.X: batch_xs,
                                              self.model.Y: batch_ys,
                                              self.model.is_training: True,
                                              self.learning_rate: self.curr_learning_rate})
        return loss, y_true, y_pred

    def train(self, sess, **kwargs):
        save_dir = kwargs.pop('save_dir', './checkpoints')

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        num_examples = self.train_set.num_examples
        num_steps_per_epoch = num_examples // self.batch_size
        steps = self.num_epochs * num_steps_per_epoch

        step_losses = []
        step_train_scores = []
        validation_scores = []

        start_time = time.time()

        for step in range(steps):

            step_loss, step_y_true, step_y_pred = self._step(sess, **kwargs)
            step_losses.append(step_loss)

            if (step + 1) % num_steps_per_epoch == 0:
                step_train_score = self.evaluator.score(step_y_true, step_y_pred)
                step_train_scores.append(step_train_score)

                if self.validation_set is not None:
                    validation_y_true = self.validation_set.labels
                    validation_y_pred = self.model.predict(sess, self.validation_set, **kwargs)
                    validation_score = self.evaluator.score(validation_y_true, validation_y_pred)
                    validation_scores.append(validation_score)
                    curr_score = validation_score
                    print('[epoch {}] loss: {:.6f} |train_score: {:.4f} |validation_score: {:.4f} |lr: {}' \
                          .format(self.curr_epoch, step_loss, step_train_score, validation_score,
                                  self.curr_learning_rate))
                else:
                    curr_score = step_train_score
                    print('[epoch {}] loss: {:.4f} |train_score: {:.4f} |lr: {}' \
                          .format(self.curr_epoch, step_loss, step_train_score,
                                  self.curr_learning_rate))

                if self.evaluator.is_better(curr=curr_score, best=self.best_score, **kwargs):
                    self.best_score = curr_score
                    self.num_idiot_epochs = 0
                    saver.save(sess, os.path.join(save_dir, 'model'))
                else:
                    self.num_idiot_epochs += 1

                self._update_learning_rate(**kwargs)
                self.curr_epoch += 1

        print('Total training time(s): {:.3f}'.format(time.time() - start_time))
        print('Best {} score: {:.3f}'.format('evaluation' if self.validation_set is not None else 'training',
                                             self.best_score))
        print('Done.')

        rst = dict()
        rst['step_losses'] = step_losses
        rst['step_train_scores'] = step_train_scores
        if self.validation_set is not None:
            rst['validation_scores'] = validation_scores

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


