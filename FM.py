#!/usr/bin/env python
#coding=utf-8
"""
Implementation of Factorization Machines
"""
import tensorflow as tf
import glob
import os
import json
from datetime import datetime
import shutil

#flag parameters
FLAGS = tf.app.flags.FLAGS
#distributed config
tf.app.flags.DEFINE_boolean("dist_mode", False, "run use distribuion mode or not")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
#model config
tf.app.flags.DEFINE_integer("embedding_szie",100,"length of latent vector")
tf.app.flags.DEFINE_boolean("first_order",True,"use first order or not")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_float("l1_reg",0.00001,"L1 regularization")
tf.app.flags.DEFINE_string("optimizer", 'Adagrad', "optimizer type")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_string("loss_type","log_loss","loss function (square_loss or log_loss)")
#train config
tf.app.flags.DEFINE_integer("feat_size",0,"Number of features")
tf.app.flags.DEFINE_integer("field_size",0,"Number of fields")
tf.app.flags.DEFINE_integer("batch_size",128,"batch size")
tf.app.flags.DEFINE_integer("num_epochs",10,"Number of epochs")
tf.app.flags.DEFINE_integer("num_threads",0,"Number of reading data threads(suggest equal to the cpu cores)")
tf.app.flags.DEFINE_string("model_root","","Specify model root directory")
tf.app.flags.DEFINE_string("cpkt_dir","","Specify cpkt directory")
tf.app.flags.DEFINE_string("out_dir","","Specify predcition directory")
tf.app.flags.DEFINE_string("train_file","","Specify train data directory")
tf.app.flags.DEFINE_string("test_file","","Specify test data directory")
tf.app.flags.DEFINE_string("va_file","","Specify validation data directory")
tf.app.flags.DEFINE_integer("num_gpus",0,"Number of gpu cores")
tf.app.flags.DEFINE_boolean("clear_existing_model",False,"whether clear existing model file")
tf.app.flags.DEFINE_integer("summary_step",500,"Save summaries every this many steps")
tf.app.flags.DEFINE_integer("log_step_count_steps",500,"The frequency, in number of global steps, that the global step/sec and the loss will be logged during training")
tf.app.flags.DEFINE_integer("save_checkpoints_steps",10000,"Save checkpoints every this many steps")
tf.app.flags.DEFINE_integer("keep_checkpoint_max",0,"The maximum number of recent checkpoint files to keep (default save all the cpkt)")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, train_eval, test, eval, export}")
tf.app.flags.DEFINE_integer("eval_step",100,"Positive number of steps for which to evaluate model")
tf.app.flags.DEFINE_integer("start_delay_secs",1000,"Start evaluating after waiting for this many seconds")
tf.app.flags.DEFINE_integer("throttle_secs",1200,"Do not re-evaluate unless the last evaluation was started at least this many seconds ago")

#input libsvm format data
#label feat_id:feat_val
#input file path support single widcard filename and list of files
def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print("Parsing ", filenames)
    def decode_libsvm(line):
        columns = tf.string_split([line]," ")
        labels = tf.string_to_number(columns.values[0],out_type=tf.float32)
        feat = tf.string_split(columns.values[1:],":")
        feat = tf.reshape(feat.values,feat.dense_shape)
        feat_id,feat_val = tf.split(feat,num_or_size_splits=2,axis=1)
        feat_id = tf.string_to_number(feat_id,out_type=tf.int32)
        feat_val = tf.string_to_number(feat_id,out_type=tf.float32)
        return  {"feat_id":feat_id,"feat_val":feat_val},labels
    
    if type(filenames) != type([]):
        filenames = glob.glob(filenames)
    
    dataset = tf.data.TextLineDataset(filenames, buffer_size=FLAGS.buffer_size_gb*1024**3)

    #dataset = dataset.shard()
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(decode_libsvm,batch_size,num_parallel_calls=FLAGS.num_threads))
    dataset = dataset.prefetch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features,batch_labels = iterator.get_next()

    return batch_features,batch_labels


def fm_fn(features,labels,mode,params):
    embd_size = params["embedding_size"]
    feat_size = params["feat_size"]
    field_size = params["field_size"] 
    
    feat_id = features["feat_id"]
    feat_id = tf.reshape(feat_id,[-1,field_size]) #None*field_size
    feat_val = features["feat_val"]
    feat_val = tf.reshape(feat_val,[-1,field_size,1]) #None*field_size

    l2_reg = tf.contrib.layers.l2_regularizer(params["l2_reg"])
    l1_reg = tf.contrib.layers.l1_regularizer(scale_l1=params["l1_reg"])
    learning_rate = params["learning_rate"]

    #featur embedding
    feat_embedding = tf.get_variable("feature_embedding",shape=[feat_size,embd_size],initializer=tf.glorot_normal_initializer(),regularizer=l2_reg)
    feat_bias = tf.get_variable("feature_bias",shape=[feat_size,1],initializer=tf.glorot_normal_initializer(),regularizer=l1_reg)
    bias = tf.get_variable("bias",shape=[1],initializer=tf.constant_initializer(0.0))
    

    with tf.variable_scope("fm"):
        emb = tf.nn.embedding_lookup(feat_embedding,feat_id) #None*field_size*embd_size
    
        sum_emb = tf.reduce_sum(tf.multiply(emb,feat_val),1) #None*embd_size
        sum_emb_square = tf.square(sum_emb)

        square_emb = tf.square(tf.multiply(emb,feat_val)) 
        square_emb_sum = tf.reduce_sum(square_emb,1) #None*embd_size
        
        wgts = tf.nn.embedding_lookup(feat_bias,feat_id) #None*field_size*1
        linear_out = tf.reduce_sum(tf.multiply(wgts,feat_val),1) #None*1
        
        bias = bias * tf.ones_like(linear_out, dtype=tf.float32)
        
        fm_out = tf.reduce_sum(0.5*tf.subtract(sum_emb_square,square_emb_sum),1) #None*1
        
        if FLAGS.first_order:
            logits = tf.add_n([fm_out,linear_out,bias])
        else:
            logits = tf.add_n([fm_out,bias])
        pred = tf.sigmoid(logits)
    
    predicted_labels = tf.cast(logits + 0.5, tf.int32)
    predictions={"prob": pred}
    eval_metric_ops = {"auc":tf.metrics.auc(labels, pred),
                       "auc_precision_recall":tf.metrics.auc(labels, pred, curve='PR'),
                       "accuracy":tf.metrics.accuracy(labels, predicted_labels),
                       "recall":tf.metrics.recall(labels, predicted_labels),
                       "precision":tf.metrics.precision(labels, predicted_labels),
                       "label/mean":tf.metrics.mean(labels)
                       }

    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}

    if FLAGS.loss_type == "square_loss":
        obj_loss = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(labels,logits)))
    if FLAGS.loss_type == "log_loss":
        obj_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    loss = obj_loss + tf.losses.get_regularization_loss()

    if FLAGS.optimizer == "Adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer == "Adagrad":
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer == "Momentum":
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == "SGD":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

def set_dist_env():
    if FLAGS.dist_mode:
        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")
        chief_hosts = worker_hosts[0:1]      #first worker is chief
        worker_hosts = worker_hosts[1:]      #rest workers are worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        if job_name == "worker" and task_index > 0:
            task_index -= 1
        
        tf_config = {'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts}, 'task': {'type': job_name, 'index': task_index }}
        print(json.dumps(tf_config))
        os.environ["TF_CONFIG"]=json.dumps(tf_config)


def main(_):
    model_dir = FLAGS.model_root
    model_dir += FLAGS.cpkt_dir if FLAGS.cpkt_dir != "" else datetime.now().strftime("%Y-%m-%d-%H")

    tr_file = FLAGS.train_file
    va_file = FLAGS.va_file
    te_file = FLAGS.test_file

    print("task type: ",FLAGS.task_type)
    print("model dir: ", model_dir)
    print("train file: ",tr_file)
    print("validation file: ",va_file)
    print("test file: ",te_file)
    print("embedding_size: ",FLAGS.embedding_size)
    print('num epochs: ', FLAGS.num_epochs)
    print("learning rate: ",FLAGS.lr)
    print("first order: ",FLAGS.first_order)
    print("optimizer: ",FLAGS.optimizer)
    print("l2 reg: ",FLAGS.l2_reg)
    print("loss type: ",FLAGS.loss_type)
    print("l1_reg: ",FLAGS.l1_reg)
    #remove if system forbide
    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % model_dir)

    set_dist_env()

    model_params = {
        "field_size": FLAGS.field_size,
        "feat_size": FLAGS.feat_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.lr,
        "l2_reg": FLAGS.l2_reg,
        "l1_reg":FLAGS.l1_reg,
        "first_order":FLAGS.first_order
    }
    if FLAGS.num_gpus != 0:
        session_config = tf.ConfigProto(per_process_gpu_memory_fraction=1.0, allow_growth=True,allow_soft_placement = True,log_device_placement=False,
        intra_op_parallelism_threads=0, inter_op_parallelism_threads=0,device_count={'GPU': FLAGS.num_gpus,"CPU":FLAGS.num_threads})
    else:
        session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads})
    
    config = tf.estimator.RunConfig().replace(save_summary_steps=FLAGS.summary_step,save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    session_config= session_config,keep_checkpoint_max=FLAGS.keep_checkpoint_max,log_step_count_steps=FLAGS.log_step_count_steps)

    FM = tf.estimator.Estimator(model_fn=fm_fn, model_dir=model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        FM.train(input_fn=lambda: input_fn(tr_file, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == "train_eval":
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_file, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(te_file, num_epochs=1, batch_size=FLAGS.batch_size),
                                            steps=FLAGS.eval_step,start_delay_secs=FLAGS.start_delay_secs,throttle_secs=FLAGS.throttle_secs)
        tf.estimator.train_and_evaluate(FM,train_spec,eval_spec)
    elif FLAGS.task_type == "test":
        FM.evaluate(input_fn=lambda: input_fn(te_file, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == "eval":
        preds = FM.predict(input_fn=lambda: input_fn(te_file, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="prob")
        with open(FLAGS.out_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_type == 'export':
        feature_spec = {
            'feat_id': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_id'),
            'feat_val': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_val')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        FM.export_savedmodel(FLAGS.out_dir, serving_input_receiver_fn)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()