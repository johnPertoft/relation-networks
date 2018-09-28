import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from rn.data import sort_of_clevr
from rn.model import model_fn


tf.logging.set_verbosity(tf.logging.INFO)


argparser = argparse.ArgumentParser()
argparser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to load.")
argparser.add_argument("--n-samples", type=int, default=100, help="Number of predictions.")
args = argparser.parse_args()

estimator = tf.estimator.Estimator(model_fn=model_fn)


# TODO: Add option to only show misclassifications.
# TODO: Add headless mode? And instead generate image report.


dataset_sample = itertools.islice(sort_of_clevr.generator(seed=99), args.n_samples)
imgs, q_encs, _, questions, answers = [list(t) for t in zip(*dataset_sample)]
input_fn = tf.estimator.inputs.numpy_input_fn({
    "img": np.stack(imgs).astype(np.float32) / 255.0,
    "question": np.stack(q_encs).astype(np.float32)},
    shuffle=False)

predictions = estimator.predict(
    input_fn=input_fn,
    checkpoint_path=args.checkpoint)

for img, question, actual_answer, predicted_answer in zip(imgs, questions, answers, predictions):
    print("Question: ", question)
    print("Predicted answer:", sort_of_clevr.Answer.decode(predicted_answer))
    print("Actual answer:", actual_answer)
    print("===============================================")

    plt.imshow(img)
    plt.draw()
    plt.waitforbuttonpress(0)
