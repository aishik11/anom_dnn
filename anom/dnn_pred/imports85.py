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
"""A dataset loader for imports85.data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf

try:
  import pandas as pd  # pylint: disable=g-import-not-at-top
except ImportError:
  pass


URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

# Order is important for the csv-readers, so we use an OrderedDict here.
defaults = collections.OrderedDict([
    
    ("transaction_id", [""]),
    ("num_var_1", [0.0]),
    ("num_var_2", [0.0]),
    ("num_var_3", [0.0]),
    ("num_var_4", [0.0]),
    ("num_var_5", [0.0]),
    ("num_var_6", [0.0]),
    ("num_var_7", [0.0]),
    ("cat_var_1", [""]),
    ("cat_var_2", [""]),
    ("cat_var_3", [""]),
    ("cat_var_4", [""]),
    ("cat_var_5", [""]),
    ("cat_var_6", [""]),
    ("cat_var_7", [""]),
    ("cat_var_8", [""]),
    ("cat_var_9", [""]),
    ("cat_var_10", [""]),
    ("cat_var_11", [""]),
    ("cat_var_12", [""]),
    ("cat_var_13", [""]),
    ("cat_var_14", [""]),
    ("cat_var_15", [""]),
    ("cat_var_16", [""]),
    ("cat_var_17", [""]),
    ("cat_var_18", [""]),
    ("cat_var_19", [0]),
    ("cat_var_20", [0]),
    ("cat_var_21", [0]),
    ("cat_var_22", [0]),
    ("cat_var_23", [0]),
    ("cat_var_24", [0]),
    ("cat_var_25", [0]),
    ("cat_var_26", [0]),
    ("cat_var_27", [0]),
    ("cat_var_28", [0]),
    ("cat_var_29", [0]),
    ("cat_var_30", [0]),
    ("cat_var_31", [0]),
    ("cat_var_32", [0]),
    ("cat_var_33", [0]),
    ("cat_var_34", [0]),
    ("cat_var_35", [0]),
    ("cat_var_36", [0]),
    ("cat_var_37", [0]),
    ("cat_var_38", [0]),
    ("cat_var_39", [0]),
    ("cat_var_40", [0]),
    ("cat_var_41", [0]),
    ("cat_var_42", [0]),
    
    
])  # pyformat: disable


types = collections.OrderedDict((key, type(value[0]))
                                for key, value in defaults.items())


def _get_imports85():
  #path = tf.contrib.keras.utils.get_file(URL.split("/")[-1], URL)
  with open("test.csv",'r') as f:
   with open("testf_new.csv",'w') as f1:
      next(f) # skip header line
      for line in f:
          f1.write(line)

  path= os.getcwd()+"\\test.csv"
  return path


def dataset(y_name="target", train_fraction=1):
  """Load the imports85 data as a (train,test) pair of `Dataset`.

  Each dataset generates (features_dict, label) pairs.

  Args:
    y_name: The name of the column to use as the label.
    train_fraction: A float, the fraction of data to use for training. The
        remainder will be used for evaluation.
  Returns:
    A (train,test) pair of `Datasets`
  """
  # Download and cache the data
  path = _get_imports85()

  # Define how the lines of the file should be parsed
  def decode_line(line):
    """Convert a csv line into a (features_dict,label) pair."""
    # Decode the line to a tuple of items based on the types of
    # csv_header.values().
    items = tf.decode_csv(line, list(defaults.values()))

    # Convert the keys and items to a dict.
    pairs = zip(defaults.keys(), items)
    features_dict = dict(pairs)

    # Remove the label from the features_dict
    #label = features_dict.pop(y_name)

    return features_dict#, label

  def has_no_question_marks(line):
    """Returns True if the line of text has no question marks."""
    # split the line into an array of characters
    chars = tf.string_split(line[tf.newaxis], "").values
    # for each character check if it is a question mark
    is_question = tf.equal(chars, "?")
    any_question = tf.reduce_any(is_question)
    no_question = ~any_question

    return no_question

  def in_training_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # If you randomly split the dataset you won't get the same split in both
    # sessions if you stop and restart training later. Also a simple
    # random split won't work with a dataset that's too big to `.cache()` as
    # we are doing here.
    num_buckets = 1000000
    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
    # Use the hash bucket id as a random number that's deterministic per example
    return bucket_id < int(train_fraction * num_buckets)

  def in_test_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # Items not in the training set are in the test set.
    # This line must use `~` instead of `not` because `not` only works on python
    # booleans but we are dealing with symbolic tensors.
    return ~in_training_set(line)

  base_dataset = (tf.contrib.data
                  # Get the lines from the file.
                  .TextLineDataset(path)
                  # drop lines with question marks.
                  .filter(has_no_question_marks))

  train = (base_dataset
           # Take only the training-set lines.
           #.filter(in_training_set)
           # Decode each line into a (features_dict, label) pair.
           .map(decode_line)
           # Cache data so you only decode the file once.
           .cache())

  # Do the same for the test-set.
  #test = (base_dataset.filter(in_test_set).cache().map(decode_line))

  return train#, test


def raw_dataframe():
  """Load the imports85 data as a pd.DataFrame."""
  # Download and cache the data
  path = _get_imports85()

  # Load it into a pandas dataframe
  df = pd.read_csv(path, names=types.keys(), dtype=types, na_values="?")

  return df


def load_data(y_name="target", train_fraction=0.7, seed=None):
  """Get the imports85 data set.

  A description of the data is available at:
    https://archive.ics.uci.edu/ml/datasets/automobile

  The data itself can be found at:
    https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

  Args:
    y_name: the column to return as the label.
    train_fraction: the fraction of the dataset to use for training.
    seed: The random seed to use when shuffling the data. `None` generates a
      unique shuffle every run.
  Returns:
    a pair of pairs where the first pair is the training data, and the second
    is the test data:
    `(x_train, y_train), (x_test, y_test) = get_imports85_dataset(...)`
    `x` contains a pandas DataFrame of features, while `y` contains the label
    array.
  """
  # Load the raw data columns.
  data = raw_dataframe()

  # Delete rows with unknowns
  data = data.dropna()

  # Shuffle the data
  np.random.seed(seed)

  # Split the data into train/test subsets.
  x_train = data.sample(frac=train_fraction, random_state=seed)
  x_test = data.drop(x_train.index)

  # Extract the label from the features dataframe.
  y_train = x_train.pop(y_name)
  y_test = x_test.pop(y_name)

  return (x_train, y_train), (x_test, y_test)
