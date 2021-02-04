# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 08:42:53 2021

@author: NguyenSon
"""
#%%
import os
import sys
import json
import time
import random
from datetime import datetime

import tensorflow as tf

import run_squad
import modeling
import tokenization
import optimization

#%%
data_dir =  'D:\jbddata'
# SQuAD json for training
train_file = 'D:/VBDI_NLP/jbddata/jbddata_train_data_copy.json'
# json for inference
predict_file = 'D:/VBDI_NLPjbddata/jbddata_dev_data_copy.json'

#%%
# init checkpoint
# notebooks_dir = '../notebooks'
working_dir = '..'
if working_dir not in sys.path:
    sys.path.append(working_dir)

init_checkpoint = 'D:/VBDI_NLP/multi_cased_L-12_H-768_A-12/bert_model.ckpt.data-00000-of-00001'

#%%
'''FINE-TUNING'''

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


# Create the output directory where all the results are saved.
output_dir = os.path.join(working_dir, 'results')
tf.compat.v1.gfile.MakeDirs(output_dir)

# The config json file corresponding to the pre-trained BERT model.
# This specifies the model architecture.
bert_config_file = 'D:/VBDI_NLP/multi_cased_L-12_H-768_A-12/bert_config.json'

# The vocabulary file that the BERT model was trained on.
vocab_file = 'D:/VBDI_NLP/multi_cased_L-12_H-768_A-12/vocab.txt'

#%%
# Whether to lower case the input text. 
# Should be True for uncased models and False for cased models.
do_lower_case = False
  
# Total batch size for predictions
predict_batch_size = 1
params = dict([('batch_size', predict_batch_size)])

# The maximum total input sequence length after WordPiece tokenization. 
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
max_seq_length = 384

# When splitting up a long document into chunks, how much stride to take between chunks.
doc_stride = 128

# The maximum number of tokens for the question. 
# Questions longer than this will be truncated to this length.
max_query_length = 64
#%%
# This is a WA to use flags from here:
flags = tf.compat.v1.flags

if 'f' not in tf.compat.v1.flags.__dict__.keys(): 
    tf.compat.v1.app.flags.DEFINE_string('f', '', 'kernel')
FLAGS = flags.FLAGS
#%%
verbose_logging = True
# Set to True if the dataset has samples with no answers. For SQuAD 1.1, this is set to False
version_2_with_negative = False

# The total number of n-best predictions to generate in the nbest_predictions.json output file.
n_best_size = 20

# The maximum length of an answer that can be generated. 
# This is needed  because the start and end predictions are not conditioned on one another.
max_answer_length = 30

# The initial learning rate for Adam
learning_rate = 5e-6

# Total batch size for training
train_batch_size = 3

# Proportion of training to perform linear learning rate warmup for
warmup_proportion = 0.1

# # Total number of training epochs to perform (results will improve if trained with epochs)
num_train_epochs = 2

global_batch_size = train_batch_size
# training_hooks = []
# training_hooks.append(run_squad.LogTrainRunHook(global_batch_size, 0))

#%%
# Validate the casing config consistency with the checkpoint name.
tokenization.validate_case_matches_checkpoint(do_lower_case, init_checkpoint)
#%%
# Create the tokenizer.
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
#%%
# Load the configuration from file
bert_config = modeling.BertConfig.from_json_file(bert_config_file)

config = tf.compat.v1.ConfigProto(log_device_placement=True) 

run_config = tf.estimator.RunConfig(
      model_dir=output_dir,
      session_config=config,
      save_checkpoints_steps=1000,
      keep_checkpoint_max=1)
#%%
# Read the training examples from the training file:
train_examples = run_squad.read_squad_examples(input_file=train_file, is_training=True)

num_train_steps = int(len(train_examples) / global_batch_size * num_train_epochs)
num_warmup_steps = int(num_train_steps * warmup_proportion)

# Pre-shuffle the input to avoid having to make a very large shuffle
# buffer in in the `input_fn`.
rng = random.Random(12345)
rng.shuffle(train_examples)

start_index = 0 
end_index = len(train_examples)
tmp_filenames = os.path.join(output_dir, "train.tf_record")

# We write to a temporary file to avoid storing very large constant tensors
# in memory.
train_writer = run_squad.FeatureWriter(
    filename=tmp_filenames,
    is_training=True)

# run_squad.convert_examples_to_features(
#     examples=train_examples[start_index:end_index],
#     tokenizer=tokenizer,
#     max_seq_length=max_seq_length,
#     doc_stride=doc_stride,
#     max_query_length=max_query_length,
#     is_training=True,
#     output_fn=train_writer.process_feature)

# train_writer.close()

#%%
tf.compat.v1.logging.info("***** Running training *****")
tf.compat.v1.logging.info("  Num orig examples = %d", end_index - start_index)
tf.compat.v1.logging.info("  Num split examples = %d", train_writer.num_features)
tf.compat.v1.logging.info("  Batch size = %d", train_batch_size)
tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
tf.compat.v1.logging.info("  LR = %f", learning_rate)

# del train_examples

def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

    tf.compat.v1.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = run_squad.create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=False)

    tvars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

      tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.compat.v1.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]

      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)

      total_loss = (start_loss + end_loss) / 2.0

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu = False)

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
      }
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params=params)

#%%
# tmp_filenames = os.path.join(output_dir, "train.tf_record")
train_input_fn = run_squad.input_fn_builder(
        input_file=train_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)

start_train = datetime.now()

estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

end_train = datetime.now()

tf.logging.info("TRAINING TIME: {}".format(end_train - start_train))
tf.logging.info("EXPORTING TRAINED MODEL")

#Export model
def serving_input_receiver_fn():
  feature_spec = {
      "input_ids" : tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
      "input_mask" : tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
      "segment_ids" : tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
      "unique_ids" :  tf.FixedLenFeature([], tf.int64)

  }
  serialized_tf_example = tf.placeholder(dtype=tf.string, 
                                        shape=[None],
                                        name='input_example_tensor')
  receiver_tensors = {'example': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

estimator.export_saved_model("D:/VBDI_NLP/saved_models/bert_base",
                              serving_input_receiver_fn=serving_input_receiver_fn)