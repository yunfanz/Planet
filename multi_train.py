import tensorflow as tf
from multi_net import RESNET, UPDATE_OPS_COLLECTION, RESNET_VARIABLES, MOVING_AVERAGE_DECAY
from image_reader import Reader, get_corpus_size
import numpy as np
import os, sys
import time
MOMENTUM = 0.9
SP2_BOX = (256,256,4)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './tmp/multi_train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', 'Data/train/',
                           """Directory where data is located""")
tf.app.flags.DEFINE_float('learning_rate', 0.002, "learning rate.")
tf.app.flags.DEFINE_float('pred_prob', 0.5, "prediction probability")
tf.app.flags.DEFINE_integer('batch_size', 8, "batch size")
tf.app.flags.DEFINE_integer('num_per_epoch', None, "max steps per epoch")
tf.app.flags.DEFINE_integer('epoch', 1, "number of epochs to train")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('is_training', True,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')
def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size
    #return num_correct
def test(sess, net, is_training, validation=False):

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    
    coord = tf.train.Coordinator()
    reader = load_images(coord, FLAGS.data_dir)
    corpus_size = reader.corpus_size
    #import IPython; IPython.embed()
    batch, _ = reader.dequeue(FLAGS.batch_size)
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)


    wlogits, mlogits = net.inference(batch)
    #loss_ = net.loss(logits, labels)
    wpredictions = tf.nn.softmax(wlogits)
    #import IPython; IPython.embed()
    #top1_error = top_k_error(predictions, labels, 1)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    #import IPython; IPython.embed()
    sess.run(init)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    latest = tf.train.latest_checkpoint(FLAGS.train_dir)
    if not latest:
        print("No checkpoint to continue from in", FLAGS.train_dir)
        sys.exit(1)
    print("resume", latest)
    saver.restore(sess, latest)

    
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)
    try:
        if FLAGS.num_per_epoch:
            batch_idx = min(FLAGS.num_per_epoch, corpus_size) // FLAGS.batch_size
        else:
            batch_idx = corpus_size // FLAGS.batch_size
        for idx in range(batch_idx):
            start_time = time.time()
            step = sess.run(global_step)

            o = sess.run(i, { is_training: True })

            loss_value = o[1]

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 1 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('Epoch %d, [%d / %d], loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (epoch, idx, batch_idx, loss_value, examples_per_sec, duration))

            if write_summary:
                summary_str = o[2]
                summary_writer.add_summary(summary_str, step)

            _, top1_error_value = sess.run([val_op, top1_error], { is_training: False })
            #pp, ll = sess.run([predictions, labels], {is_training:False})
            #print('Predictions: ', pp)
            #print('labels: ', ll)
            print('Validation top1 error %.2f' % top1_error_value)

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
        #G
    finally:
        print('Finished, output see {}'.format(FLAGS.train_dir))
        coord.request_stop()
        coord.join(threads)
def _scoring(tp, fp, fn):
    p = tp/(tp + fp)
    r = tp/(tp + fn)
    b = 2.
    if p ==0. and r ==0.:
        return 0
    else:
        return (1 + b**2)*(p*r)/(b**2*p + r)

def get_m_score(mlogits, labels):
    tp, fn, fp = 0, 0, 0
    for i in range(13):
        mpred = tf.cast(tf.round(tf.nn.softmax(mlogits[:,i])), tf.int32)
        #import IPython; IPython.embed()
        tp += tf.reduce_sum(tf.mul(mpred[:,1], labels[:,i+1]))
        fp += tf.reduce_sum(tf.mul(mpred[:,1], 1 - labels[:,i+1]))
        fn += tf.reduce_sum(tf.mul(1 - mpred[:,1], labels[:,i+1]))
        # tp += tf.tensordot(mpred[:,1], labels[:,i+1])
        # fp += tf.tensordot(mpred[:,1], 1 - labels[:,i+1])
        # fn += tf.tensordot(1 - mpred[:,1], labels[:,i+1])
    tp = tf.cast(tp, tf.float32)
    fp = tf.cast(fp, tf.float32)
    fn = tf.cast(fn, tf.float32)

    return _scoring(tp, fp, fn)

def train(sess, net, is_training, keep_prob):

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    
    coord = tf.train.Coordinator()
    reader = load_images(coord, FLAGS.data_dir)
    corpus_size = reader.corpus_size
    #import IPython; IPython.embed()
    train_batch, labels = reader.dequeue(FLAGS.batch_size)
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    
    logits_weather, logits_multi = net.inference(train_batch)
    #import IPython; IPython.embed() 
    wloss_ = net.loss(logits_weather, labels[:,0], name='weather_loss')
    mloss_ = tf.add_n([net.loss(logits_multi[:,i], labels[:,i+1], name='multi_loss'+str(i)) for i in range(13)])
    loss_ = wloss_ + 0.08 * mloss_
    predictions = tf.nn.softmax(logits_weather)
    #import IPython; IPython.embed()
    top1_error = top_k_error(predictions, labels[:,0], 1)
    m_score = get_m_score(logits_multi, labels)


    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([wloss_]))
    tf.scalar_summary('wloss_avg', ema.average(wloss_))
    
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([mloss_]))
    tf.scalar_summary('mloss_avg', ema.average(mloss_))
    # loss_avg
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.scalar_summary('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.99, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]), ema.apply([m_score]))
    top1_error_avg = ema.average(top1_error)
    m_score_avg = ema.average(m_score)
    tf.scalar_summary('val_top1_error_avg', top1_error_avg)
    tf.scalar_summary('m_score_avg', m_score_avg)

    
    ###
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    tf.scalar_summary('learning_rate', learning_rate)
    #opt = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
    #opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    ###
    grads = opt.compute_gradients(loss_)
    #wgrads = opt.compute_gradients(wloss_) #no need to separate, tensorflow knows
    #mgrads = opt.compute_gradients(0.05 * mloss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        #tf.image_summary('images', images)

        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    #import IPython; IPython.embed()
    sess.run(init)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not latest:
            print("No checkpoint to continue from in", FLAGS.train_dir)
            sys.exit(1)
        print("resume", latest)
        saver.restore(sess, latest)

    
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)
    #import IPython; IPython.embed()
    try:
        for epoch in range(FLAGS.epoch):
            if epoch % 8 == 0 and epoch > 1:
                FLAGS.learning_rate /=  10. 
            if FLAGS.num_per_epoch:
                batch_idx = min(FLAGS.num_per_epoch, corpus_size) // FLAGS.batch_size
            else:
                batch_idx = corpus_size // FLAGS.batch_size
            for idx in range(batch_idx):
                start_time = time.time()

                step = sess.run(global_step)
                i = [train_op, loss_]

                write_summary = step % 100 and step > 1
                if write_summary:
                    i.append(summary_op)

                o = sess.run(i, { is_training: True, keep_prob: 0.5, learning_rate: FLAGS.learning_rate })
                #import IPython; IPython.embed()

                loss_value = o[1]

                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    examples_per_sec = FLAGS.batch_size / float(duration)
                    format_str = ('Epoch %d, [%d / %d], loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (epoch, idx, batch_idx, loss_value, examples_per_sec, duration))

                if write_summary:
                    summary_str = o[2]
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step > 1 and step % 1000 == 0:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)

                # Run validation periodically
                if step > 1 and step % 100 == 0:
                    _, top1_error_value, mscore_value = sess.run([val_op, top1_error, m_score], 
                        { is_training: False, keep_prob: 1})
                    #pp, ll = sess.run([predictions, labels], {is_training:False})
                    #print('Predictions: ', pp)
                    #print('labels: ', ll)
                    print('weather top1 error {}, multi_score {}'.format(top1_error_value, mscore_value))

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
        #G
    finally:
        print('Finished, output see {}'.format(FLAGS.train_dir))
        coord.request_stop()
        coord.join(threads)

def get_predictions(weather_pred, mpred, weathers, classes, batched=True):

    if batched:
        #print(weather_pred.shape, mpred.shape)
        string_list = []
        for n in range(weather_pred.shape[0]):
            string_list.append(get_predictions(weather_pred[n], mpred[n], weathers, classes, batched=False))
        return string_list
    else:
        label_str= weathers[np.argmax(weather_pred)] + ' '
        for i in range(len(classes)):
            prediction = mpred[i]
            if prediction[1]> FLAGS.pred_prob:
                label_str += classes[i] + ' '
        labels = sorted(label_str.split(' '))
        label_str = ' '.join(labels)
        return label_str.strip()

def predict(sess, net, is_training, keep_prob, prefix='test_', append=False):

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    fname = FLAGS.train_dir+"/results_"+str(FLAGS.pred_prob)+".csv"
    if not append:
        outfile = open(fname, 'w')
        outfile.write("image_name,tags\n")
        outfile.close()
    
    
    coord = tf.train.Coordinator()
    reader = load_images(coord, FLAGS.data_dir, train=False)
    corpus_size = reader.corpus_size
    #import IPython; IPython.embed()
    test_batch, img_id = reader.dequeue(FLAGS.batch_size)
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    wlogits, mlogits = net.inference(test_batch)
    wpred = tf.nn.softmax(wlogits)
    #mpred = [tf.nn.softmax(mlogits[:, i]) for i in range(13)]
    mpred = tf.nn.softmax(mlogits)
    weathers = ["cloudy", "partly_cloudy", "haze", "clear"]
    classes = ["primary", "agriculture", "water", "road", "cultivation", "habitation", "bare_ground", 
                "slash_burn", "conventional_mine", "artisinal_mine", "selective_logging", 
                "blooming", "blow_down"]
    # label_str = get_predictions(wpred, mpred, weathers, classes)

    
    init = tf.initialize_all_variables()
    #import IPython; IPython.embed()
    sess.run(init)
    saver = tf.train.Saver(tf.all_variables())
    latest = tf.train.latest_checkpoint(FLAGS.train_dir)
    if not latest:
        print("No checkpoint to continue from in", FLAGS.train_dir)
        sys.exit(1)
    print("resume", latest)
    saver.restore(sess, latest)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    outfile = open(fname, 'a')
    sample_cnt = 0
    try:
        while True:
            if sample_cnt >= corpus_size-1:
                break
            add_cnt = FLAGS.batch_size
            weather_scores, multi_scores, image_id = sess.run([wpred, mpred, img_id], { is_training: False, keep_prob: 1 })
            #import IPython; IPython.embed()
            #multi_scores = np.swapaxes(np.asarray(multi_scores),0,1)
            multi_scores = np.asarray(multi_scores)
            string_list = get_predictions(weather_scores, multi_scores, weathers, classes, batched=True)
            for n, label_str in enumerate(string_list):
                #print(prefix+str(image_id[n])+','+label_str)
                if n > 1:
                    if image_id[n] < image_id[n-1]:
                        add_cnt = n
                        break
                outfile.write(prefix+str(image_id[n])+','+label_str+'\n')
            sample_cnt += add_cnt


            if sample_cnt % 200 == 0:
                print("{}/{}".format(sample_cnt, corpus_size))
    finally:
        print('Finished, output see {}'.format(fname))
        coord.request_stop()
        coord.join(threads)
        outfile.close()


def load_images(coord, data_dir, train=True):
    if not data_dir:
        data_dir = './Data/train/'


    reader = Reader(
        data_dir,
        coord,
        pattern='*.tif',
        queue_size=FLAGS.batch_size*8, 
        min_after_dequeue=FLAGS.batch_size,
        q_shape=SP2_BOX, 
        n_threads=1,
        train=train)

    print('Using data_dir{}, size {}'.format(data_dir, reader.corpus_size))
    return reader


def main(_):

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = False
    sess = tf.Session(config=sessconfig)
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    
    is_training = tf.placeholder('bool', [], name='is_training')
    keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
    # for resnet 101: num_blocks=[3, 4, 23, 3]
    # for resnet 152: num_blocks=[3, 8, 36, 3]
    resnet50 = RESNET(sess, 
                 dim=2,
                 num_weather=4,
                 num_classes=13,
                 num_blocks=[3, 4, 6, 3],  # first chan is not a block
                 num_chans=[32,32,64,128,256],
                 use_bias=False, # defaults to using batch norm
                 bottleneck=True,
                 is_training=is_training)

    #WRN = RESNET(sess, 
    #            dim=2,
    #            num_weather=4,
    #            num_classes=13,
    #            num_blocks=[1, 2, 3, 1],  # first chan is not a block
    #            num_chans=[128,128,256,512,1024],
    #            use_bias=False, # defaults to using batch norm
    #            bottleneck=False,
    #            is_training=is_training)

    # net = RESNET(sess, 
    #             dim=2,
    #             num_weather=4,
    #             num_classes=13,
    #             num_blocks=[3, 4, 4, 3],  # first chan is not a block
    #             num_chans=[16,16,32,64,128],
    #             use_bias=False, # defaults to using batch norm
    #             bottleneck=True,
    #             is_training=is_training)
    net = resnet50
    if FLAGS.is_training:
        train(sess, net, is_training, keep_prob)
    else:
        if FLAGS.data_dir.endswith("test") or FLAGS.data_dir.endswith("test/"):
            predict(sess, net, is_training, keep_prob, prefix='test_', append=False)
        else:
            predict(sess, net, is_training, keep_prob, prefix='file_', append=True)


if __name__ == '__main__':
    tf.app.run()
