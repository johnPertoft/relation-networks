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
argparser.add_argument("--n-samples", type=int, default=300, help="Number of predictions.")
argparser.add_argument("--seed", type=int, default=99, help="Seed for random generator for sort of clevr.")
argparser.add_argument("--only-misclassifications", action="store_true",
                       help="Whether to only show misclassifications.")
args = argparser.parse_args()

estimator = tf.estimator.Estimator(model_fn=model_fn)


# TODO: Add headless mode? And instead generate image report.


dataset_sample = itertools.islice(sort_of_clevr.generator(seed=args.seed), args.n_samples)
imgs, q_encs, _, questions, answers = [list(t) for t in zip(*dataset_sample)]
input_fn = tf.estimator.inputs.numpy_input_fn({
    "img": np.stack(imgs).astype(np.float32) / 255.0,
    "question": np.stack(q_encs).astype(np.float32)},
    shuffle=False)

predictions = estimator.predict(
    input_fn=input_fn,
    checkpoint_path=args.checkpoint)


for img, question, actual_answer, predicted_answer in zip(imgs, questions, answers, predictions):
    answer = sort_of_clevr.Answer.decode(predicted_answer)

    if args.only_misclassifications and str(answer) == actual_answer:
        continue

    print("Question: ", question)
    print("Predicted answer:", answer)
    print("Actual answer:", actual_answer)
    print("===============================================")

    plt.imshow(img)
    plt.draw()
    plt.waitforbuttonpress(0)
