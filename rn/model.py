import functools

import tensorflow as tf


def cnn_model_fn(x, is_training):
    conv = functools.partial(tf.layers.conv2d, padding="same")
    norm = functools.partial(tf.layers.batch_normalization, training=is_training)
    activation = tf.nn.relu

    net = x
    net = activation(norm(conv(net, 32, 3, 2)))
    net = activation(norm(conv(net, 64, 3, 2)))
    net = activation(norm(conv(net, 128, 3, 2)))
    net = conv(net, 256, 3, 2)

    return net


def relation_model_fn(c, is_training):
    activation = tf.nn.relu

    net = c
    net = activation(tf.layers.dense(net, 2000))
    net = activation(tf.layers.dense(net, 2000))
    net = activation(tf.layers.dense(net, 2000))
    net = activation(tf.layers.dense(net, 2000))

    return net


def answer_model_fn(r, answer_dim, is_training):
    activation = tf.nn.relu

    net = r
    net = activation(tf.layers.dense(net, 2000))
    net = activation(tf.layers.dense(net, 1000))
    net = activation(tf.layers.dense(net, 500))
    net = activation(tf.layers.dense(net, 100))
    net = tf.layers.dense(net, answer_dim)

    return net


def model_fn(features, labels, mode, params, config):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    cnn = tf.make_template("cnn", cnn_model_fn, is_training=is_training)
    relation = tf.make_template("relation", relation_model_fn, is_training=is_training)
    answer = tf.make_template("answer", answer_model_fn, answer_dim=labels.shape[1], is_training=is_training)

    # The "objects" are the spatial coordinates of the cnn output.
    objects = cnn(features["img"])
    _, h, w, c = objects.get_shape().as_list()
    n_objects = h * w
    objects = tf.reshape(objects, [-1, n_objects, c])

    # Cartesian product of objects extracted from one image concatenated pairwise.
    object_pairs_indices = tf.convert_to_tensor([(i, j) for i in range(n_objects) for j in range(n_objects) if i != j])
    object_pairs = tf.concat((
        tf.gather(objects, object_pairs_indices[:, 0], axis=1),
        tf.gather(objects, object_pairs_indices[:, 1], axis=1)),
        axis=2)

    # Relation model constructed from each object pair concatenated with the question representation.
    n_pairs = object_pairs.shape[1]
    question = features["question"]
    question = question[:, tf.newaxis, :]
    question = tf.tile(question, [1, n_pairs, 1])
    relation_input = tf.concat((
        object_pairs,
        question),
        axis=2)
    combined_relations = tf.reduce_sum(relation(relation_input), axis=1)

    # TODO: Test that the object pairing part works as expected.

    logits = answer(combined_relations)

    predictions = None
    if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
        probabilities = tf.nn.softmax(logits)
        predictions = tf.one_hot(tf.argmax(probabilities, axis=1), depth=probabilities.shape[1])

    loss = None
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        tf.losses.add_loss(tf.losses.softmax_cross_entropy(labels, logits))
        loss = tf.losses.get_total_loss()

    eval_metric_ops = None
    evaluation_hooks = None
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)
        }

    train_op = None
    training_hooks = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(loss, global_step=global_step)
        # TODO: Add summaries to train hooks?
            # Include text summaries with question paired with actual and predicted answer.

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=training_hooks,
        eval_metric_ops=eval_metric_ops,
        evaluation_hooks=evaluation_hooks)
