import argparse

import tensorflow as tf

from rn.data import sort_of_clevr
from rn.model import model_fn


tf.logging.set_verbosity(tf.logging.INFO)


argparser = argparse.ArgumentParser()
argparser.add_argument("--model-dir", type=str, help="Model directory for training and evaluation output.")
argparser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
args = argparser.parse_args()

hparams = {}

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=args.model_dir,
    params=hparams)

tf.estimator.train_and_evaluate(
    estimator=estimator,
    train_spec=tf.estimator.TrainSpec(
        input_fn=lambda: sort_of_clevr.tf_dataset(args.batch_size, seed=0),
        hooks=None,
        max_steps=300000),
    eval_spec=tf.estimator.EvalSpec(
        input_fn=lambda: sort_of_clevr.tf_dataset(args.batch_size, seed=1)))

# TODO: Pick best checkpoint.
estimator.evaluate(
    input_fn=lambda: sort_of_clevr.tf_dataset(args.batch_size, seed=2))
