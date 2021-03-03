# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:16:59 2021

@author: NguyenSon
"""

import tensorflow.compat.v1 as tf
import os
import numpy as np

import run_squad_cnn_inf
import tokenization
import modeling


# GET CONFIG AND VOCAB
bert_config_file = r'D:\VBDI_NLP\multi_cased_L-12_H-768_A-12/bert_config.json'
# "The config json file corresponding to the pre-trained BERT model. "
# "This specifies the model architecture."

vocab_file = 'D:\VBDI_NLP\multi_cased_L-12_H-768_A-12/vocab.txt'
# "The vocabulary file that the BERT model was trained on."

output_dir = 'D:\VBDI_NLP\bert\tmp\demo'
# "The output directory where the model checkpoints will be written."

## Other parameters
predict_file = r'D:\VBDI_NLP\jbddata\jbddata_test_data_copy.json'

init_checkpoint = r'D:\VBDI_NLP\bert\tmp\squad_basecnn_jp\model.ckpt-4967'
# "Initial checkpoint (usually from a pre-trained BERT model)."

do_lower_case = True
# "Whether to lower case the input text. Should be True for uncased "
# "models and False for cased models."

max_seq_length = 384
# "The maximum total input sequence length after WordPiece tokenization. "
# "Sequences longer than this will be truncated, and sequences shorter "
# "than this will be padded.")

doc_stride = 128
# "When splitting up a long document into chunks, how much stride to "
# "take between chunks.")

max_query_length = 64
# "The maximum number of tokens for the question. Questions longer than "
# "this will be truncated to this length.")

train_batch_size = 32
# "Total batch size for training.")

predict_batch_size = 8
# "Total batch size for predictions.")

learning_rate = 5e-5
# "The initial learning rate for Adam.")

num_train_epochs = 3.0
# "Total number of training epochs to perform.")

warmup_proportion = 0.1
# "Proportion of training to perform linear learning rate warmup for. "
# "E.g., 0.1 = 10% of training.")

save_checkpoints_steps = 1000
# "How often to save the model checkpoint.")

iterations_per_loop = 1000
# "How many steps to make in each estimator call.")

n_best_size = 20
# "The total number of n-best predictions to generate in the "
# "nbest_predictions.json output file.")

max_answer_length = 30
# "The maximum length of an answer that can be generated. This is needed "
# "because the start and end predictions are not conditioned on one another.")

tf.logging.set_verbosity(tf.logging.INFO)
working_dir = r'.\output_result'
# Create the output directory where all the results are saved.
output_dir = os.path.join(working_dir, 'results')
tf.gfile.MakeDirs(output_dir)

# Total batch size for predictions
predict_batch_size = 8
params = dict([('batch_size', predict_batch_size)])

# This is a WA to use flags from here:
flags = tf.flags

if 'f' not in tf.flags.FLAGS:
    tf.app.flags.DEFINE_string('f', '', 'kernel')
FLAGS = flags.FLAGS


#%%
# Validate the casing config consistency with the checkpoint name.
tokenization.validate_case_matches_checkpoint(do_lower_case, init_checkpoint)

# Create the tokenizer.
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

# Load the configuration from file
bert_config = modeling.BertConfig.from_json_file(bert_config_file)

def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = run_squad_cnn_inf.create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=False)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    output_spec = None
    predictions = {"unique_ids": unique_ids,
                   "start_logits": start_logits,
                   "end_logits": end_logits}
    output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    return output_spec

config = tf.ConfigProto(log_device_placement=True)

run_config = tf.estimator.RunConfig(model_dir=None,
                                    session_config=config,
                                    save_checkpoints_steps=1000,
                                    keep_checkpoint_max=1)

estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   config=run_config,
                                   params=params)

#%%

def extractor(predict_file):
    tf.logging.info('==========================BEGIN LOADING FILE====================================')
    eval_examples = run_squad_cnn_inf.read_squad_examples(
            input_file=predict_file, is_training=False)
    
    eval_writer = run_squad_cnn_inf.FeatureWriter(
        filename=os.path.join(output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []
    tf.logging.info('==========================APPEND FEATURE========================================')
    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)
    
    run_squad_cnn_inf.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()
    
    tf.logging.info("********************* RUNNING PREDICTIONS ***********************")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", predict_batch_size)
    
    all_results = []
    
    tf.logging.info('==========================INPUT FUNCTION BUILDER==================================')
    predict_input_fn = run_squad_cnn_inf.input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=False)
    
    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    for result in estimator.predict(
        predict_input_fn, yield_single_examples=True):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      all_results.append(
          run_squad_cnn_inf.RawResult(
              unique_id=unique_id,
              start_logits=start_logits,
              end_logits=end_logits))
    
    output_prediction_file = os.path.join(output_dir, "predictions.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(output_dir, "null_odds.json")
    
    tf.logging.info('========================== WRITING PREDICTION TO FILE ========================================')
    run_squad_cnn_inf.write_predictions(eval_examples, eval_features, all_results,
                      n_best_size, max_answer_length,
                      do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file)
    
    tf.logging.info('========================== FINISH EXTRACTING INFOMATION========================================')