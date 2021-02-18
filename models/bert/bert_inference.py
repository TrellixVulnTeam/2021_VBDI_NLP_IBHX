# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:42:34 2021

@author: NguyenSon
"""

'''CONFIG'''
from BERT_NVIDIA import run_squad
import json
import tensorflow as tf
from BERT_NVIDIA import modeling
from BERT_NVIDIA import tokenization
import time
import random
import os

tf.logging.set_verbosity(tf.logging.INFO)
working_dir = './output_result'
# Create the output directory where all the results are saved.
output_dir = os.path.join(working_dir, 'results')
tf.gfile.MakeDirs(output_dir)

# The config json file corresponding to the pre-trained BERT model.
# This specifies the model architecture.
bert_config_file = os.path.join('./multi_cased_L-12_H-768_A-12/bert_config.json')

# The vocabulary file that the BERT model was trained on.
vocab_file = os.path.join('./multi_cased_L-12_H-768_A-12/vocab.txt')

# # Depending on the mixed precision flag we use different fine-tuned model
# if use_mixed_precision_model:
#     init_checkpoint = os.path.join(data_dir, 'finetuned_model_fp16/model.ckpt-8144')
# else:
#     init_checkpoint = os.path.join(data_dir, 'finetuned_model_fp32/model.ckpt-8144')


init_checkpoint = os.path.join(data_dir, 'finetuned_model_fp32/model.ckpt-8144')
# Whether to lower case the input text.
# Should be True for uncased models and False for cased models.
do_lower_case = True

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

# This is a WA to use flags from here:
flags = tf.flags

if 'f' not in tf.flags.FLAGS:
    tf.app.flags.DEFINE_string('f', '', 'kernel')
FLAGS = flags.FLAGS

# The total number of n-best predictions to generate in the nbest_predictions.json output file.
n_best_size = 20

# The maximum length of an answer that can be generated.
# This is needed  because the start and end predictions are not conditioned on one another.
max_answer_length = 30
#%%
'''TOKIENIZER & CREATE MODEL'''
# Validate the casing config consistency with the checkpoint name.
tokenization.validate_case_matches_checkpoint(do_lower_case, init_checkpoint)

# Create the tokenizer.
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

# Load the configuration from file
bert_config = modeling.BertConfig.from_json_file(bert_config_file)

def model_fn(features, labels, mode, params):# pylint: disable=unused-argument
    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    (start_logits, end_logits) = run_squad.create_model(
                                            bert_config=bert_config,
                                            is_training=False,
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
'''INFERENCE'''
input_file = 
eval_examples = run_squad.read_squad_examples(input_file=input_file,
                                              is_training=False)

eval_writer = run_squad.FeatureWriter(filename=os.path.join(output_dir, "eval.tf_record"),
                                      is_training=False)

eval_features = []
def append_feature(feature):
    eval_features.append(feature)
    eval_writer.process_feature(feature)


# Loads a data file into a list of InputBatch's
run_squad.convert_examples_to_features(examples=eval_examples,
                                        tokenizer=tokenizer,
                                        max_seq_length=max_seq_length,
                                        doc_stride=doc_stride,
                                        max_query_length=max_query_length,
                                        is_training=False,
                                        output_fn=append_feature)

eval_writer.close()

tf.logging.info("***** Running predictions *****")
tf.logging.info("  Num orig examples = %d", len(eval_examples))
tf.logging.info("  Num split examples = %d", len(eval_features))
tf.logging.info("  Batch size = %d", predict_batch_size)

predict_input_fn = run_squad.input_fn_builder(input_file=eval_writer.filename,
                                              batch_size=predict_batch_size,
                                              seq_length=max_seq_length,
                                              is_training=False,
                                              drop_remainder=False)

all_results = []
eval_hooks = [run_squad.LogEvalRunHook(predict_batch_size)]
eval_start_time = time.time()
for result in estimator.predict(predict_input_fn, 
                                yield_single_examples=True, 
                                hooks=eval_hooks, 
                                checkpoint_path=init_checkpoint):
    
    unique_id = int(result["unique_ids"])
    start_logits = [float(x) for x in result["start_logits"].flat]
    end_logits = [float(x) for x in result["end_logits"].flat]
    all_results.append(run_squad.RawResult(unique_id=unique_id,
                                           start_logits=start_logits,
                                           end_logits=end_logits))

eval_time_elapsed = time.time() - eval_start_time

eval_time_wo_startup = eval_hooks[-1].total_time
num_sentences = eval_hooks[-1].count * predict_batch_size
avg_sentences_per_second = num_sentences * 1.0 / eval_time_wo_startup

tf.logging.info("-----------------------------")
tf.logging.info("Total Inference Time = {%0.2f} Inference Time W/O start up overhead = {%0.2f} \
                Sentences processed = {%d}".format(eval_time_elapsed, eval_time_wo_startup,num_sentences))
tf.logging.info("Inference Performance = {%0.4f} sentences/sec".format(avg_sentences_per_second))
tf.logging.info("-----------------------------")

output_prediction_file = os.path.join(output_dir, "predictions.json")
output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
output_null_log_odds_file = os.path.join(output_dir, "null_odds.json")

run_squad.write_predictions(eval_examples, eval_features, all_results,
                            n_best_size, max_answer_length,
                            do_lower_case, output_prediction_file,
                            output_nbest_file, output_null_log_odds_file)

tf.logging.info("Inference Results:")

# Here we show only the prediction results, nbest prediction is also available in the output directory
results = ""
with open(output_prediction_file, 'r') as json_file:
    data = json.load(json_file)
    for question in eval_examples:
        results += "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(question.qas_id, question.question_text, data[question.qas_id])


from IPython.display import display, HTML
display(HTML("<table><tr><th>Id</th><th>Question</th><th>Answer</th></tr>{}</table>".format(results)))