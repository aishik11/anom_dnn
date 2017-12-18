# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Regression using the DNNRegressor Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import imports85  # pylint: disable=g-bad-import-order

STEPS = 5000
PRICE_NORM_FACTOR = 1000


def main(argv):
  """Builds, trains, and evaluates the model."""
  assert len(argv) == 1
  (train, test) = imports85.dataset()

  # Switch the labels to units of thousands for better convergence.
  def normalize_price(features, labels):
    return features, labels #/ PRICE_NORM_FACTOR

  train = train.map(normalize_price)
  test = test.map(normalize_price)

  # Build the training input_fn.
  def input_train():
    return (
        # Shuffling with a buffer larger than the data set ensures
        # that the examples are well mixed.
        train.shuffle(1000).batch(128)
        # Repeat forever
        .repeat().make_one_shot_iterator().get_next())

  # Build the validation input_fn.
  def input_test():
    return (test.shuffle(1000).batch(128)
            .make_one_shot_iterator().get_next())

  # The first way assigns a unique weight to each category. To do this you must
  # specify the category's vocabulary (values outside this specification will
  # receive a weight of zero). Here we specify the vocabulary using a list of
  # options. The vocabulary can also be specified with a vocabulary file (using
  # `categorical_column_with_vocabulary_file`). For features covering a
  # range of positive integers use `categorical_column_with_identity`.


  cat_var_1=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_1", hash_bucket_size=1000)
  cat_var_2=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_2", hash_bucket_size=1000)
  cat_var_4=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_4", hash_bucket_size=1000)
  cat_var_5=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_5", hash_bucket_size=1000)
  cat_var_6=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_6", hash_bucket_size=1000)
  cat_var_9=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_9", hash_bucket_size=1000)
  cat_var_10=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_10", hash_bucket_size=1000)
  cat_var_11=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_11", hash_bucket_size=1000)
  cat_var_12=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_12", hash_bucket_size=1000)
  cat_var_13=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_13", hash_bucket_size=1000)
  cat_var_14=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_14", hash_bucket_size=1000)
  cat_var_15=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_15", hash_bucket_size=1000)
  cat_var_16=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_16", hash_bucket_size=1000)
  cat_var_17=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_17", hash_bucket_size=1000)
  cat_var_18=tf.feature_column.categorical_column_with_hash_bucket(
      key="cat_var_18", hash_bucket_size=1000)



  feature_columns = [
      tf.feature_column.numeric_column(key="num_var_1"),
      tf.feature_column.numeric_column(key="num_var_2"),
      tf.feature_column.numeric_column(key="num_var_4"),
      tf.feature_column.numeric_column(key="num_var_5"),
      tf.feature_column.numeric_column(key="num_var_6"),
      tf.feature_column.numeric_column(key="num_var_7"),
      # Since this is a DNN model, convert categorical columns from sparse
      # to dense.
      # Wrap them in an `indicator_column` to create a
      # one-hot vector from the input.
      #tf.feature_column.indicator_column(body_style),
      # Or use an `embedding_column` to create a trainable vector for each
      # index.
      tf.feature_column.indicator_column(cat_var_1),
      tf.feature_column.indicator_column(cat_var_2),
      tf.feature_column.indicator_column(cat_var_4),
      tf.feature_column.indicator_column(cat_var_5),
      tf.feature_column.indicator_column(cat_var_6),
      tf.feature_column.indicator_column(cat_var_9),
      tf.feature_column.indicator_column(cat_var_10),
      tf.feature_column.indicator_column(cat_var_11),
      tf.feature_column.indicator_column(cat_var_12),
      tf.feature_column.indicator_column(cat_var_13),
      tf.feature_column.indicator_column(cat_var_14),
      tf.feature_column.indicator_column(cat_var_15),
      tf.feature_column.indicator_column(cat_var_16),
      tf.feature_column.indicator_column(cat_var_17),
      tf.feature_column.indicator_column(cat_var_18),
      #tf.feature_column.embedding_column(make, dimension=3),
  ]

  # Build a DNNRegressor, with 2x20-unit hidden layers, with the feature columns
  # defined above as input.
  model = tf.estimator.DNNRegressor(
      hidden_units=[20, 20], feature_columns=feature_columns)

  # Train the model.
  model.train(input_fn=input_train, steps=STEPS)

  # Evaluate how the model performs on data it has not yet seen.
  eval_result = model.evaluate(input_fn=input_test)

  # The evaluation returns a Python dictionary. The "average_loss" key holds the
  # Mean Squared Error (MSE).
  average_loss = eval_result["average_loss"]

  # Convert MSE to Root Mean Square Error (RMSE).
  print("\n" + 80 * "*")
  print("\nRMS error for the test set: ${:.0f}"
        .format(PRICE_NORM_FACTOR * average_loss**0.5))

  print()


if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
