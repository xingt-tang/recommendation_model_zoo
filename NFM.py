#!/usr/bin/env python
#coding=utf-8
"""
Implementation of Neural Factorization Machines
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

#model parameters
tf.app.flags.DEFINE_string("keep_prob",'0.5,0.6,0.8',"keep probability for bilinear layer and hidden layer(1-dropout_ratio),last one is bilinear layer dropout")
tf.app.flags.DEFINE_string("hidden_layer",'256,64',"size of each hidden layers")
tf.app.flags.DEFINE_float("lr",0.001,"learning rate")
tf.app.flags.DEFINE_float("embedding_size",128,"Size of embedding")
tf.app.flags.DEFINE_float("l2_reg",0.0,"Regularizer for DNN")
tf.app.flags.DEFINE_float("l2_bilinear",0.0,"L2 regularizer for bilinear part")
tf.app.flags.DEFINE_float("l1_bilinear",0.0,"L1 regularizer for bilinear part")
tf.app.flags.DEFINE_string("optimizer","Adagrad","Optimizer type:{Adam,Adagrad,SGD,Momentum}")
tf.app.flags.DEFINE_boolean("batch_norm",True,"use batch_normailization or not")
tf.app.flags.DEFINE_float("batch_norm_decay",0.9,"decay for moving average")
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


def nfm_fn(features,labels,mode,params):
    embd_size = params["embedding_size"]
    feat_size = params["feat_size"]
    field_size = params["field_size"] 
    
    feat_id = features["feat_id"]
    feat_id = tf.reshape(feat_id,[-1,field_size]) #None*field_size
    feat_val = features["feat_val"]
    feat_val = tf.reshape(feat_val,[-1,field_size,1]) #None*field_size

    keep_prob = list(map(float,params["keep_prob"].split(",")))
    assert len(keep_prob) >= 1
    hidden_layer = list(map(int,params["hidden_layer"].split(",")))
    assert len(keep_prob) == len(hidden_layer)+1
    
    learning_rate = params["learning_rate"]
    l2_reg= params["l2_reg"]
    l2_bilinear = params["l2_bilinear"]
    l1_bilinear = params["l1_bilinear"]
    l1_l2_reg = tf.contrib.layers.l1_l2_regularizer(scale_l1=l1_bilinear,scale_l2=l2_bilinear)
    #featur embedding
    feat_embedding = tf.get_variable("feature_embedding",shape=[feat_size,embd_size],initializer=tf.glorot_normal_initializer(),regularizer=l1_l2_reg)
    feat_bias = tf.get_variable("feature_bias",shape=[feat_size,1],initializer=tf.glorot_normal_initializer(),regularizer=l1_l2_reg)
    bias = tf.get_variable("bias",shape=[1],initializer=tf.constant_initializer(0.0))

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_phase = True
    else:
        train_phase = False
    
    #bilinear part
    with tf.variable_scope("bilinear"):
        emb = tf.nn.embedding_lookup(feat_embedding,feat_id) #None*field_size*embd_size
        
        sum_emb = tf.reduce_sum(tf.multiply(emb,feat_val),1) #None*embd_size
        sum_emb_square = tf.square(sum_emb)

        square_emb = tf.square(tf.multiply(emb,feat_val)) 
        square_emb_sum = tf.reduce_sum(square_emb,1) #None*embd_size

        bilinear = 0.5*tf.subtract(sum_emb_square,square_emb_sum) #None*embd_size
        
        if params["batch_norm"]:
            bilinear = batch_norm_layer(bilinear,train_phase,"bilinear_bn")
        if train_phase:
            bilinear = tf.nn.dropout(bilinear,keep_prob[-1])
    
    with tf.variable_scope("deep"):
        deep_out = bilinear
        
        for i in range(len(hidden_layer)):
            deep_out = tf.contrib.layers.fully_connected(inputs=deep_out, num_outputs=hidden_layer[i],weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp_%d' % i)
            
            if params["batch_norm"]:
                deep_out = batch_norm_layer(deep_out,train_phase,scope_bn = "bn_%d"%i)
            
            if train_phase:
                deep_out = tf.nn.dropout(deep_out,keep_prob[i])
        
        deep_out = tf.contrib.layers.fully_connected(inputs=deep_out, num_outputs = 1, activation_fn=tf.identity,
        weights_regularizer = tf.contrib.layers.l2_regularizer(l2_reg), scope="prediction") 
        
        deep_out = tf.reshape(deep_out,shape=[-1])  #None*1

    with tf.variable_scope("linear"):
        wgts = tf.nn.embedding_lookup(feat_bias,feat_id) #None*field_size*1
        linear_out = tf.reduce_sum(tf.multiply(wgts,feat_val),1) #None*1
    
    with tf.variable_scope("nfm_out"):
        bias = bias * tf.ones_like(linear_out, dtype=tf.float32)
        logits = tf.add_n([linear_out,bias,deep_out])
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


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


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
    print("batch normalization: ", FLAGS.batch_norm)
    print("batch normalization decay: ", FLAGS.batch_norm_decay)
    print("hidden layers: ",FLAGS.hidden_layer)
    print("keep probability: ",FLAGS.keep_prob)
    print("optimizer: ",FLAGS.optimizer)
    print("l2 reg: ",FLAGS.l2_reg)
    print("loss type: ",FLAGS.loss_type)
    print("l2_bilinear: ",FLAGS.l2_bilinear)
    print("l1_bilinear: ",FLAGS.l1_bilinear)
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
        "batch_norm": FLAGS.batch_norm,
        "l2_reg": FLAGS.l2_reg,
        "hidden_layer": FLAGS.hidden_layer,
        "keep_prob": FLAGS.keep_prob,
        "l2_bilinear":FLAGS.l2_bilinear,
        "l1_bilinear":FLAGS.l1_bilinear
    }
    if FLAGS.num_gpus != 0:
        session_config = tf.ConfigProto(per_process_gpu_memory_fraction=1.0, allow_growth=True,allow_soft_placement = True,log_device_placement=False,
        intra_op_parallelism_threads=0, inter_op_parallelism_threads=0,device_count={'GPU': FLAGS.num_gpus,"CPU":FLAGS.num_threads})
    else:
        session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads})
    
    config = tf.estimator.RunConfig().replace(save_summary_steps=FLAGS.summary_step,save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    session_config= session_config,keep_checkpoint_max=FLAGS.keep_checkpoint_max,log_step_count_steps=FLAGS.log_step_count_steps)

    NFM = tf.estimator.Estimator(model_fn=nfm_fn, model_dir=model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        NFM.train(input_fn=lambda: input_fn(tr_file, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == "train_eval":
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_file, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(te_file, num_epochs=1, batch_size=FLAGS.batch_size),
                                            steps=FLAGS.eval_step,start_delay_secs=FLAGS.start_delay_secs,throttle_secs=FLAGS.throttle_secs)
        tf.estimator.train_and_evaluate(NFM,train_spec,eval_spec)
    elif FLAGS.task_type == "test":
        NFM.evaluate(input_fn=lambda: input_fn(te_file, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == "eval":
        preds = NFM.predict(input_fn=lambda: input_fn(te_file, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="prob")
        with open(FLAGS.out_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_type == 'export':
        feature_spec = {
            'feat_id': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_id'),
            'feat_val': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_val')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        NFM.export_savedmodel(FLAGS.out_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

 