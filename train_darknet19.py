import os
import numpy as np
import tensorflow as tf
from datasets import asirra
from models.nn import Darknet19
from train.evaluators import AccuracyEvaluator
from train.optimizer import AdamOptimizer


dataset = asirra.read_trainval_sets('./data/dogs-vs-cats', validation_size=2500, one_hot=True)

save_dir = './checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

hp_d = dict()
hp_d['save_dir'] = save_dir

hp_d['batch_size'] = 64
hp_d['num_epochs'] = 160

# TODO: data augmentation routines
hp_d['augment_train'] = False
hp_d['augment_test'] = False

hp_d['initial_learning_rate'] = 0.001
hp_d['learning_rate_patience'] = 30
hp_d['learning_rate_decay'] = 0.1
hp_d['learning_rate_eps'] = 1e-8

hp_d['momentum'] = 0.9
hp_d['weight_decay'] = 0.0005

hp_d['accuracy_eps'] = 1e-4

print('Training set stats:')
print('shape={}'.format(dataset.train.images.shape))
print('cat={}, dog={}'.format((dataset.train.labels[:, 1] == 0).sum(), (dataset.train.labels[:, 1] == 1).sum()))

if dataset.validation is None:
    print('Validation set is none.')
else:
    print('Validation set stats:')
    print('\tshape={}'.format(dataset.validation.images.shape))
    print('\tmin={}, max={}'.format(dataset.validation.images.min(), dataset.validation.images.max()))
    print('\tcat={}, dog={}'.format((dataset.validation.labels[:, 1] == 0).sum(),
                                    (dataset.validation.labels[:, 1] == 1).sum()))

graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(graph=graph, config=config)

model = Darknet19([224, 224, 3], 2, **hp_d)
evaluator = AccuracyEvaluator()
optimizer = AdamOptimizer(model=model,
                          train_set=dataset.train,
                          validation_set=dataset.validation,
                          evaluator=evaluator,
                          **hp_d)
optimizer.train(sess, **hp_d)
