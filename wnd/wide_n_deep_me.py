#!/usr/bin/env python
#coding=utf-8
import tensorflow as tf
from datetime import date,timedelta
import glob
import json
import os
"""
Rewrite wide and deep models, implemented by tensorflow low-level API
Support batch normalization/l1_l2 regularization/dropout/continue feature
"""

#flag parameters
FLAGS = tf.app.flags.FLAGS

# resources config
tf.app.flags.DEFINE_boolean("dist_mode", False, "run use distribuion mode or not")
tf.app.flags.DEFINE_boolean("batch_norm",True,"use batch_normailization or not")
tf.app.flags.DEFINE_float("batch_norm_decay",0.9,"decay for moving average")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 10, "Number of threads")

# model hyper-params config
tf.app.flags.DEFINE_integer("feature_size", 0, "Number of features")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.app.flags.DEFINE_integer("wide_size", 0, "wide size")
tf.app.flags.DEFINE_integer("wide_field", 0, "Number of wide fields")
tf.app.flags.DEFINE_integer("deep_c_field", 939, "deep continous part size")
tf.app.flags.DEFINE_integer("deep_e_field", 9, "deep discrete part size")
tf.app.flags.DEFINE_integer("deep_e_size",0,"deep embedding size")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")

# logs and data config
tf.app.flags.DEFINE_integer("buffer_size_gb", 3, "dataset buffer size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_integer("throttle_secs", 600, "evaluate every 10mins")
tf.app.flags.DEFINE_float("dnn_learning_rate", 0.0005, "dnn_learning rate")
tf.app.flags.DEFINE_float("linear_learning_rate", 0.0005, "linear_learning rate")
tf.app.flags.DEFINE_string("dropout", '0.5', "dropout")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_float("l1_reg",0.0001,"L1 regularization for wide part")
tf.app.flags.DEFINE_string("linear_optimizer", 'Ftrl', "linear optimizer type {Ftrl, Adagrad, SGD}")
tf.app.flags.DEFINE_float("beta",1.0,"beta parameters for Ftrl")
tf.app.flags.DEFINE_string("dnn_optimizer",'Adagrad',"dnn optimizer type {Adagrad,Sgd,Momentum,Adam}")
tf.app.flags.DEFINE_string("data_file", '', "data dir")
tf.app.flags.DEFINE_string("test_file", '', "test dir")
tf.app.flags.DEFINE_string("te_file",'',"test file")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, predict, export}")
tf.app.flags.DEFINE_string("model_type", 'deep', "model type {'wide', 'deep', 'wide_n_deep'}")

#libsvm format
#label deep_embedding_index deep_continue_index wide_index
def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print("Parsing", filenames)
    def decode_libsvm(line):
        columns = tf.string_split([line],' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        #TODO:modify
        assert len(columns.values)-1 == FLAGS.deep_e_field + FLAGS.deep_c_field + FLAGS.wide_field 
        
        deep_ebd_part = tf.string_split(columns.values[1:FLAGS.deep_e_field+1],':')
        deep_con_part = tf.string_split(columns.values[FLAGS.deep_e_field+1:FLAGS.deep_e_field+FLAGS.deep_c_field+1],':')
        wide_part = tf.string_split(columns.values[FLAGS.deep_e_field+FLAGS.deep_c_field+1:],':')
        #parse deep embedding part
        deep_ebd_id_vals = tf.reshape(deep_ebd_part.values, deep_ebd_part.dense_shape)
        deep_ebd_feat_ids,deep_ebd_feat_vals = tf.split(deep_ebd_id_vals, num_or_size_splits=2, axis=1)
        deep_ebd_feat_ids = tf.string_to_number(deep_ebd_feat_ids, out_type=tf.int32)
        deep_ebd_feat_vals = tf.string_to_number(deep_ebd_feat_vals, out_type=tf.float32)
        #parse deep continue part
        deep_con_id_vals = tf.reshape(deep_con_part.values, deep_con_part.dense_shape)
        deep_con_feat_ids,deep_con_feat_vals = tf.split(deep_con_id_vals, num_or_size_splits=2, axis=1)
        deep_con_feat_ids = tf.string_to_number(deep_con_feat_ids,out_type=tf.int32)
        deep_con_feat_vals = tf.string_to_number(deep_con_feat_vals, out_type=tf.float32)
        #parse wide part
        wide_id_vals = tf.reshape(wide_part.values, wide_part.dense_shape)
        wide_feat_ids,wide_feat_vals = tf.split(wide_id_vals,num_or_size_splits=2, axis=1)
        wide_feat_ids = tf.string_to_number(wide_feat_ids,out_type=tf.int32)
        wide_feat_vals = tf.string_to_number(wide_feat_vals,out_type=tf.float32)
        
        return {"deep_ebd_feat_ids":deep_ebd_feat_ids,"deep_ebd_feat_vals":deep_ebd_feat_vals,"deep_con_feat_ids":deep_con_feat_ids,
        "deep_con_feat_vals":deep_con_feat_vals,"wide_feat_ids":wide_feat_ids,"wide_feat_vals":wide_feat_vals},labels
    
    if type(filenames) != type([]):
        filenames = glob.glob(filenames)
    dataset = tf.data.TextLineDataset(filenames, buffer_size=FLAGS.buffer_size_gb*1024**3).map(decode_libsvm,
    num_parallel_calls=FLAGS.num_threads).prefetch(batch_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features,batch_labels = iterator.get_next()

    return batch_features,batch_labels

def wide_deep_model_fn(features,labels,mode,params):

    if 'wide' in FLAGS.model_type:
        #----wide-part-data
        wide_ids = features["wide_feat_ids"]
        wide_vals = features["wide_feat_vals"] #None*num
        wide_size = FLAGS.wide_size
    if 'deep' in FLAGS.model_type:
        #----deep-part-continous-data
        deep_c_vals = features["deep_con_feat_vals"]
        deep_c_field = FLAGS.deep_c_field
        #----deep-part-embedding-data
        #deep_e_vals = features["deep_ebd_feat_vals"]
        deep_e_ids = features["deep_ebd_feat_ids"]
        deep_e_field = FLAGS.deep_e_field
        deep_e_size = FLAGS.deep_e_size
        embedding_size = FLAGS.embedding_size
    
    l1_l2_reg = tf.contrib.layers.l1_l2_regularizer(scale_l1=FLAGS.l1_reg,scale_l2=FLAGS.l2_reg)
    l2_reg = tf.contrib.layers.l2_regularizer(FLAGS.l2_reg)
    #----wide scope
    if 'wide' in FLAGS.model_type:
        with tf.variable_scope("wide"):
            wide_w = tf.get_variable(name = 'wide_w',shape=[wide_size],initializer=tf.glorot_normal_initializer(),regularizer=l1_l2_reg)
            wide_wgts = tf.nn.embedding_lookup(wide_w, wide_ids) #None*num
            wide_logit = tf.reduce_sum(tf.multiply(wide_wgts,wide_vals),1,keepdims=True)
    #----network params
    layers = [int(x) for x in FLAGS.deep_layers.split(',')]
    dropout = [float(x) for x in FLAGS.dropout.split(',')]
    assert len(layers) == len(dropout) or len(dropout) == 1
    if len(dropout) == 1:
        dropout *= len(layers)
    
    #----deep scope
    if 'deep' in FLAGS.model_type:
        with tf.variable_scope("deep"):
            deep_w = tf.get_variable(name = 'deep_w', shape=[deep_e_size, embedding_size], initializer=tf.glorot_uniform_initializer(),regularizer=l2_reg)
            deep_c_wgts = deep_c_vals
            deep_c_wgts = tf.reshape(deep_c_vals,shape=[-1, deep_c_field])
            deep_e_wgts = tf.nn.embedding_lookup(deep_w, deep_e_ids)
            #print (deep_e_wgts)
            deep_e_wgts = tf.reshape(deep_e_wgts, shape=[-1, deep_e_field*embedding_size])
            deep_inputs = tf.concat((deep_c_wgts, deep_e_wgts), axis=1)

            if FLAGS.batch_norm:
                if mode == tf.estimator.ModeKeys.TRAIN:
                    train_phase = True
                else:
                    train_phase = False
            else:
                normalizer_fn = None
                normalizer_params = None
            
            for i in range(len(layers)):
                deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], activation_fn = tf.nn.relu,
                    weights_regularizer=l2_reg, scope="mlp_{}".format(i))
                
                if FLAGS.batch_norm:
                    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_{}'.format(i)) 

                if mode == tf.estimator.ModeKeys.TRAIN:
                    deep_inputs = tf.nn.dropout(deep_inputs, keep_prob = dropout[i])
            
            y_logit = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn = tf.identity,
                weights_regularizer = l2_reg)
            deep_logit = tf.identity(y_logit)

    with tf.variable_scope("wide_n_deep"):
        if FLAGS.model_type == 'wide_n_deep':
            logits = wide_logit + deep_logit
        elif FLAGS.model_type == 'deep':
            logits = deep_logit
        else:
            logits = wide_logit
        pred = tf.sigmoid(logits)

    #print(pred.shape, labels.shape)
    prediction = {"prob": pred}
    #ftrl need logloss excluding regularization loss
    reg_loss = tf.losses.get_regularization_loss(scope="deep") if FLAGS.linear_optimizer == "Ftrl" else tf.losses.get_regularization_loss()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)) +reg_loss


    if FLAGS.dnn_optimizer == 'Adagrad':
        #dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.dnn_learning_rate, initial_accumulator_value=1e-8)
        dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.dnn_learning_rate)
    elif FLAGS.dnn_optimizer == 'Momentum':
        dnn_optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.dnn_learning_rate, momentum=0.95)
    elif FLAGS.dnn_optimizer == 'Adam':
        dnn_optimizer = tf.train.AdamOptimizer(learning_rate= FLAGS.dnn_learning_rate)
    elif FLAGS.dnn_optimizer == 'Sgd':
        dnn_optimizer = tf.train.GradientDescentOptimizer(learning_rate = FLAGS.dnn_learning_rate)
    
    if FLAGS.linear_optimizer == "Ftrl":
        linear_optimizer = tf.train.FtrlOptimizer(FLAGS.linear_learning_rate,initial_accumulator_value=FLAGS.beta,
        l1_regularization_strength=FLAGS.l1_reg,l2_regularization_strength=FLAGS.l2_reg)
    elif FLAGS.linear_optimizer == "Adagrad":
        linear_optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.linear_learning_rate)
    elif FLAGS.linear_optimizer == "Sgd":
        linear_optimizer = tf.train.GradientDescentOptimizer(learning_rate = FLAGS.linear_learning_rate)

    train_ops = []
    if 'deep' in FLAGS.model_type:
        train_ops.append(dnn_optimizer.minimize(loss, global_step=tf.train.get_global_step(), var_list=
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="deep")))
    if 'wide' in FLAGS.model_type:
        train_ops.append(linear_optimizer.minimize(loss, var_list=
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="wide")))
    train_op = tf.group(*train_ops)


    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction, loss=loss, train_op=train_op)

    predicted_classes = tf.cast(logits + 0.5, tf.int32)
    #print (predicted_classes)
    eval_metric_ops = {"auc":tf.metrics.auc(labels, pred),
                       "auc_precision_recall":tf.metrics.auc(labels, pred, curve='PR'),
                       "accuracy":tf.metrics.accuracy(labels, predicted_classes),
                       "recall":tf.metrics.recall(labels, predicted_classes),
                       "precision":tf.metrics.precision(labels, predicted_classes),
                       "label/mean":tf.metrics.mean(labels)
                       }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction, loss=loss, eval_metric_ops=eval_metric_ops)

    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:tf.estimator.export.PredictOutput(prediction)}
    if mode ==  tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction, export_outputs=export_outputs)

def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z

def set_dist_env():
    if FLAGS.dist_mode:
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1] # get first worker as chief
        worker_hosts = worker_hosts[1:] # the rest as worker
        print('ps_host', ps_hosts)
        print('worker_host', worker_hosts)
        print('chief_hosts', chief_hosts)
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
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

        tf_config = {'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts}, 'task': {'type': job_name, 'index': task_index }}
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main(_):

    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir
    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_file ', FLAGS.data_file)
    print('test_file ', FLAGS.test_file)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('feature_size ', FLAGS.feature_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('wide_size',FLAGS.wide_size)
    print('wide_field',FLAGS.wide_field)
    print('deep_c_field',FLAGS.deep_c_field)
    print('deep_e_size',FLAGS.deep_e_size)
    print('deep_e_field',FLAGS.deep_e_field)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('batch_norm_decay',FLAGS.batch_norm_decay)
    print('dnn_optimizer',FLAGS.dnn_optimizer)
    print('linear_optimizer ', FLAGS.linear_optimizer)
    print('dnn_learning_rate ', FLAGS.dnn_learning_rate)
    print('linear_learning_rate ', FLAGS.linear_learning_rate)
    print('l2_reg ', FLAGS.l2_reg)
    print('l1_reg', FLAGS.l1_reg)

    #------init Envs------
    tr_files = FLAGS.data_file
    te_files = FLAGS.test_file

    #------bulid Tasks-----
    set_dist_env()
    #config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
    #        log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(intra_op_parallelism_threads=64, inter_op_parallelism_threads=64),
                                                save_checkpoints_steps=3000,
                                                keep_checkpoint_max=50,
                                                log_step_count_steps=FLAGS.log_steps,
                                                save_summary_steps=FLAGS.log_steps)
    # build model
    wide_n_deep = tf.estimator.Estimator(model_fn=wide_deep_model_fn, model_dir=FLAGS.model_dir, config=config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size),
                                            #steps=5000,             # 5k steps about 10-part in each eval
                                            start_delay_secs=300,  # start eval after 5min when begin traininng
                                            throttle_secs=300)     # evaluate every 5min for wide
        tf.estimator.train_and_evaluate(wide_n_deep, train_spec, eval_spec)
        wide_n_deep.train(input_fn=lambda: input_fn(tr_files))
    elif FLAGS.task_type == 'eval':
        preds = wide_n_deep.evaluate(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size))
        print("Metric is {}".format(preds))
    elif FLAGS.task_type == 'infer':
        preds = wide_n_deep.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="prob")
        with open(tr_files+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_type == 'export':
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        wide_n_deep.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
