#!/usr/bin/env python
#coding=utf-8
"""
TensorFlow Implementation of <<Deep & Cross Network for Ad Click Predictions>> with the fellowing features：
#1 Input pipline using Dataset high level API, Support parallel and prefetch reading
#2 Train pipline using Coustom Estimator by rewriting model_fn
#3 Support distincted training using TF_CONFIG
#4 Support export_model for TensorFlow Serving
#5 Support real value feature
"""

import shutil
import os
import json
import glob
from datetime import date, timedelta
from time import time
import random
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
#distributed params
tf.app.flags.DEFINE_boolean("dist_mode", False, "distribuion mode {0-loacal, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 16, "Number of threads")
#input params
tf.app.flags.DEFINE_integer("cat_feature_size", 0, "Number of category features")
tf.app.flags.DEFINE_integer("rel_feature_size", 0, "Number of real feature fields")
tf.app.flags.DEFINE_integer("cat_field_size", 0, "Number of category fields")
tf.app.flags.DEFINE_integer("embedding_size", 128, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 3, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 128, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_integer("cross_layers", 3, "cross layers, polynomial degree")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '', "data dir")
tf.app.flags.DEFINE_string("data_file", '', "data dir")
tf.app.flags.DEFINE_string("test_file", '',"test file")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")
tf.app.flags.DEFINE_integer("buffer_size_gb", 3, "dataset buffer size")

#libsvm format
#label embedding_part continue_part
def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)
    def decode_libsvm(line):
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        #TODO:modify data format according to platform.
        assert len(columns.values)-1 == FLAGS.cat_field_size + FLAGS.rel_feature_size
        
        real_part = tf.string_split(columns.values[FLAGS.cat_field_size+1:],':')
        cat_part = tf.string_split(columns.values[1:FLAGS.cat_field_size+1], ':')
        real_id_vals = tf.reshape(real_part.values,real_part.dense_shape)
        real_feat_ids, real_feat_vals = tf.split(real_id_vals,num_or_size_splits=2,axis=1)
        real_feat_ids = tf.string_to_number(real_feat_ids, out_type=tf.int32)
        real_feat_vals = tf.string_to_number(real_feat_vals, out_type=tf.float32)
        cat_id_vals = tf.reshape(cat_part.values,cat_part.dense_shape)
        cat_feat_ids,cat_feat_vals = tf.split(cat_id_vals,num_or_size_splits=2,axis=1)
        cat_feat_ids = tf.string_to_number(cat_feat_ids,out_type=tf.int32)
        cat_feat_vals = tf.string_to_number(cat_feat_vals,out_type=tf.float32)
        return {"cat_feat_ids": cat_feat_ids, "cat_feat_vals": cat_feat_vals,"real_feat_vals":real_feat_vals}, labels
    
    if type(filenames) != type([]):
        filenames = glob.glob(filenames)
    
    dataset = tf.data.TextLineDataset(filenames, buffer_size=FLAGS.buffer_size_gb * 1024 ** 3).\
        map(decode_libsvm, num_parallel_calls=FLAGS.num_threads).prefetch(batch_size)    

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) 

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def model_fn(features, labels, mode, params):
    #------hyperparameters----
    field_size = params["field_size"]
    cat_feature_size = params["cat_feature_size"]
    rel_feature_size = params["rel_feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    deep_layers  = list(map(int, params["deep_layers"].split(',')))
    cross_layers = params["cross_layers"]
    dropout = list(map(float, params["dropout"].split(',')))


    #------bulid weights------
    l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
    Cross_B = tf.get_variable(name='cross_b', shape=[cross_layers, field_size*embedding_size + rel_feature_size], initializer=tf.glorot_normal_initializer())
    Cross_W = tf.get_variable(name='cross_w', shape=[cross_layers, field_size*embedding_size + rel_feature_size], initializer=tf.glorot_normal_initializer(),regularizer=l2_regularizer)
    Feat_Emb = tf.get_variable(name='emb', shape=[cat_feature_size, embedding_size], initializer=tf.glorot_normal_initializer())

    #------build category feaure-------
    cat_feat_ids  = features['cat_feat_ids']
    cat_feat_ids = tf.reshape(cat_feat_ids,shape=[-1,field_size])
    cat_feat_vals = features['cat_feat_vals']
    cat_feat_vals = tf.reshape(cat_feat_vals,shape=[-1,field_size])

    #------build real feature---------
    rel_feat_vals = features['real_feat_vals']
    rel_feat_vals = tf.reshape(rel_feat_vals,shape=[-1,rel_feature_size])

    #------build f(x)------
    with tf.variable_scope("Embedding-layer"):
        embeddings = tf.nn.embedding_lookup(Feat_Emb, cat_feat_ids) 		    # None * F * K
        feat_vals = tf.reshape(cat_feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals) 				    # None * F * K
        x0 = tf.reshape(embeddings,shape=[-1,field_size*embedding_size])    # None * (F*K)
        x0 = tf.concat([rel_feat_vals,x0],axis=1)
        x0 = tf.reshape(x0,shape=[-1,field_size*embedding_size + rel_feature_size])

    with tf.variable_scope("Cross-Network"):
        xl = x0
        for l in range(cross_layers):
            wl = tf.reshape(Cross_W[l],shape=[-1,1])                        # (F*K) * 1
            xlw = tf.matmul(xl, wl)                                         # None * 1
            xl = x0 * xlw + xl + Cross_B[l]                                 # None * (F*K) broadcast

    with tf.variable_scope("Deep-Network"):
        if FLAGS.batch_norm:
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
            else:
                train_phase = False
            normalizer_fn = None
            normalizer_params = None

        x_deep = x0
        for i in range(len(deep_layers)):
            x_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=deep_layers[i], scope='mlp%d' % i)
            if FLAGS.batch_norm:
                x_deep = batch_norm_layer(x_deep, train_phase=train_phase, scope_bn='bn_%d' %i)   
            if mode == tf.estimator.ModeKeys.TRAIN:
                x_deep = tf.nn.dropout(x_deep, keep_prob=dropout[i])                             

    with tf.variable_scope("DCN-out"):
        x_stack = tf.concat([xl, x_deep], 1)	# None * ( F*K+ deep_layers[i])
        y = tf.contrib.layers.fully_connected(inputs=x_stack, num_outputs=1, activation_fn=tf.identity,scope='out_layer')
        y = tf.reshape(y,shape=[-1])
        pred = tf.sigmoid(y)

    predictions={"prob": pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    #------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + tf.losses.get_regularization_loss()

    # Provide an estimator spec for `ModeKeys.EVAL`
    predicted_classes = tf.cast(pred + 0.5, tf.int32)
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred),
        "pr-auc":tf.metrics.auc(labels,pred,curve="PR"),
        "accuracy":tf.metrics.accuracy(labels, predicted_classes),
        "recall":tf.metrics.recall(labels, predicted_classes),
        "precision":tf.metrics.precision(labels, predicted_classes),
        "label/mean":tf.metrics.mean(labels)

    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    #------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'Sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    #return tf.estimator.EstimatorSpec(
    #        mode=mode,
    #        loss=loss,
    #        train_op=train_op,
    #        predictions={"prob": pred},
    #        eval_metric_ops=eval_metric_ops)

def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z

def set_dist_env():
    if FLAGS.dist_mode:      # 集群分布式模式
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1] # get first worker as chief
        worker_hosts = worker_hosts[1:] # the rest as worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('worker_host', worker_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        # if job_name == "worker" and task_index == 1:
        #     job_name = 'evaluator'
        #     task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 0:
            task_index -= 1

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main(_):
    #------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir
    #FLAGS.data_dir  = FLAGS.data_dir + FLAGS.dt_dir

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('cat_feature_size ', FLAGS.cat_feature_size)
    print('rel_feature_size', FLAGS.rel_feature_size)
    print('field_size ', FLAGS.cat_field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('cross_layers ', FLAGS.cross_layers)
    print('dropout ', FLAGS.dropout)
    #print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('batch_norm_decay ', FLAGS.batch_norm_decay)
    print('batch_norm ', FLAGS.batch_norm)
    print('l2_reg ', FLAGS.l2_reg)

    #------init Envs------
#    tr_files = glob.glob("%s/tr*libsvm" % FLAGS.data_dir)
#    random.shuffle(tr_files)
#    print("tr_files:", tr_files)
#    va_files = glob.glob("%s/va*libsvm" % FLAGS.data_dir)
#    print("va_files:", va_files)
#    te_files = glob.glob("%s/te*libsvm" % FLAGS.data_dir)
#    print("te_files:", te_files)
    tr_files = FLAGS.data_file
    te_files = FLAGS.test_file
    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    set_dist_env()

    #------bulid Tasks------
    model_params = {
        "field_size": FLAGS.cat_field_size,
        "cat_feature_size": FLAGS.cat_feature_size,
        "rel_feature_size": FLAGS.rel_feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "batch_norm_decay": FLAGS.batch_norm_decay,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "cross_layers": FLAGS.cross_layers,
        "dropout": FLAGS.dropout
    }
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
    session_config = tf.ConfigProto(gpu_options=gpu_options,
                                    allow_soft_placement=True, log_device_placement=False,
                                    intra_op_parallelism_threads=0, inter_op_parallelism_threads=0,
                                    device_count={'GPU': 2})
    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
            log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    config = tf.estimator.RunConfig().replace(
        session_config=session_config,
        save_checkpoints_steps=25000, keep_checkpoint_max=50, log_step_count_steps=FLAGS.log_steps,
        save_summary_steps=FLAGS.log_steps)

    Estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        if FLAGS.dist_mode:
            train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
            eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size),
                                            steps=1000,
                                            start_delay_secs=1000,
                                            throttle_secs=1200)
            tf.estimator.train_and_evaluate(Estimator, train_spec, eval_spec)
        else:
            Estimator.train(input_fn=lambda:input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))

    elif FLAGS.task_type == 'eval':
        Estimator.evaluate(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'infer':
        preds = Estimator.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size),predict_keys="prob")
        with open(FLAGS.data_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_type == 'export':
        #feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        #feature_spec = {
        #    'feat_ids': tf.FixedLenFeature(dtype=tf.int64, shape=[None, FLAGS.field_size]),
        #    'feat_vals': tf.FixedLenFeature(dtype=tf.float32, shape=[None, FLAGS.field_size])
        #}
        #serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        Estimator.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
