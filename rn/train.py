import argparse

import tensorflow as tf

from rn.data import sort_of_clevr
from rn.model import model_fn


argparser = argparse.ArgumentParser()
argparser.add_argument("--model-dir")
args = argparser.parse_args()

hparams = {}

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=args.model_dir,
    params=hparams)


ds = tf.data.Dataset.from_generator(
    lambda: sort_of_clevr.generator(False),
    output_types=(tf.uint8, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([sort_of_clevr.IMG_SIZE, sort_of_clevr.IMG_SIZE, 3]),
                   tf.TensorShape([sort_of_clevr.Questions.dim]),
                   tf.TensorShape([sort_of_clevr.Answer.dim])))
ds = ds.shuffle(1024)
ds = ds.batch(32)


def input_fn():
    img, q, a = ds.make_one_shot_iterator().get_next()
    return {"img": tf.cast(img, tf.float32) / 255.0, "question": q}, a


train_spec = tf.estimator.TrainSpec(
    input_fn=input_fn,
    hooks=None,
    max_steps=None
)

eval_spec = tf.estimator.EvalSpec(
    input_fn=input_fn)  # TODO: Temp input fn.

tf.estimator.train_and_evaluate(
    estimator=estimator,
    train_spec=train_spec,
    eval_spec=eval_spec)

# TODO: Final test set evaluation.
