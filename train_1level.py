import tensorflow as tf
from utils import *
from model_level1 import Level1Model
from beam_search import CaptionGenerator
import config


def main():
    model = Level1Model(config, mode='training')

    # idx = h5py.File('./data/val/val_idx.h5')['labels']
    data = h5py.File('./data/train/train_caption.h5')
    images = data['images']
    captions = data['first_layer_labels']
    caption_idx = data['first_layer_label2imgid']


    # val_data = h5py.File('./data/val/val_caption.h5')
    # val_images = data['images']
    # val_captions = data['first_layer_labels']
    # val_caption_idx = data['first_layer_label2imgid']

    # first_level_label_start_ix = data['first_layer_label_start_ix']
    # first_level_label_end_ix = data['first_layer_label_end_ix']
    # second_level_label_start_ix = data['label_start_ix']
    # second_level_label_end_ix = data['label_end_ix']
    # second_level_label_pos = data['label_position']
    # second_level_labels = data['labels']

    optimizer = tf.train.AdamOptimizer
    log_path = './model/level1_test/'
    pretrained_model = None

    generator = CaptionGenerator(model, model.level1_word2ix, None,
                                 beam_size_1level=3, beam_size_2level=None,
                                 encourage_1level=0.0, encourage_2level=None,
                                 level2=False)
    loss = model.build()
    n_examples = caption_idx.shape[0]
    # n_examples_val = val_caption_idx.shape[0]
    n_iters_per_epoch = int(np.ceil(float(n_examples) / config.batch_size))
    # n_iters_val = int(np.ceil(float(n_examples_val) / config.batch_size))
    # print [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    print [v for v in tf.trainable_variables() if v.name.startswith("resnet/block7/bottleneck22/b/batch_normalization/")]
    test1 = [v for v in tf.trainable_variables() if v.name == ("resnet/block7/bottleneck22/b/batch_normalization/beta:0")][0]
    test2 = [v for v in tf.trainable_variables() if v.name == ("resnet/block7/bottleneck22/b/batch_normalization/gamma:0")][0]
    # test3 = [v for v in  tf.get_default_graph().as_graph_def().node if v.name == ("resnet/block7/bottleneck22/b/batch_normalization/moving_mean")][0]
    print test1, test2
    with tf.name_scope('optimizer'):
        optimizer = optimizer(learning_rate=0.0000004)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='level1')
            if config.train_resnet:
                optim_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet')
            level1_grads = tf.gradients(loss, optim_vars)
            grads_and_vars = [(i, j) for i, j in zip(level1_grads, optim_vars) if i is not None]
            grads_and_vars = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grads_and_vars]
            # # todo: here check the batch-norm moving average/var
            # if config.train_resnet:
            #     optim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet')
            #     resnet_grads = tf.gradients(model.resnet.features, optim_vars)
            #     resnet_pairs = [(i, j) for i, j in zip(resnet_grads, optim_vars) if i is not None]
            #     grads_and_vars.extend(resnet_pairs)

            # batchnorm_updates = tf.get_collection('resnet_update_ops')
            # batchnorm_updates_op = tf.group(*batchnorm_updates)
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
            # train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    # summary op
    print '************************'
    tf.summary.scalar('batch_loss', loss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    # for grad, var in grads_and_vars:
    #     tf.summary.histogram(var.op.name + '/gradient', grad)

    summary_op = tf.summary.merge_all()

    print "The number of epoch: %d" % config.n_epochs
    print "Data size: %d" % n_examples
    print "Batch size: %d" % config.batch_size
    print "Iterations per epoch: %d" % n_iters_per_epoch

    config_ = tf.ConfigProto(allow_soft_placement=True)
    config_.gpu_options.per_process_gpu_memory_fraction=0.9
    config_.gpu_options.allow_growth = True
    with tf.Session(config=config_) as sess:
        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=40)

        if pretrained_model is not None:
            print "Start training with pretrained Model.."
            saver.restore(sess, pretrained_model)

        prev_loss = -1
        curr_loss = 0
        start_t = time.time()
        i_global = 0
        for e in range(config.n_epochs):
            rand_idxs = list(np.random.permutation(n_examples))
            for i in range(n_iters_per_epoch):
                i_global += 1
                rand_idx = sorted(rand_idxs[i * config.batch_size:(i + 1) * config.batch_size])
                captions_batch = captions[rand_idx]
                img_idx = list(caption_idx[rand_idx])
                # print img_idx
                img_batch = crop_image(images[img_idx], True)
                # print decode_captions(captions_batch, model.level1_model.idx_to_word)
                # img_feature = sess.run(model.resnet.features, {model.resnet.images: img_batch})
                feed_dict = {model.level1_model.captions: captions_batch,
                             model.level1_model.resnet.images: img_batch}
                _, l = sess.run([train_op, loss], feed_dict)
                # print 'batch norm beta:', sess.run(test1)[:10]
                # print 'batch norm gamma:', sess.run(test2)[:10]
                # print 'batch norm moving ave:', sess.run('resnet/block7/bottleneck22/b/batch_normalization/moving_mean:0')[:10]

                # l = sess.run(loss, feed_dict)
                curr_loss += l
                # write summary for tensorboard visualization
                if i % 1000 == 0:
                    summary = sess.run(summary_op, feed_dict)
                    summary_writer.add_summary(summary, e * n_iters_per_epoch + i)

                if (i + 1) % config.print_every == 0:
                    print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" % (e + 1, i + 1, l)
                    # print img_idx, caption_idx == img_idx[0]
                    ground_truths = captions_batch[0]
                    decoded = decode_captions(ground_truths, model.level1_model.idx_to_word)
                    for j, gt in enumerate(decoded):
                        print "Ground truth %d: %s" % (j + 1, gt)
                        print ground_truths
                    predicted = generator.beam_search(sess, img_batch[0:1,:,:,:])
                    decoded_predict = decode_captions(np.asarray(predicted), model.level1_model.idx_to_word)
                    print "Generated caption: %s\n" % decoded_predict
                    print predicted
                    print '***************'
                if (i_global + 1) % 1000 == 0:
                    saver.save(sess, os.path.join('./model', 'model_level1_trained_bn'), global_step=i_global + 1)
                    print "model-%s saved." % (i_global + 1)

            print "Previous epoch loss: ", prev_loss
            print "Current epoch loss: ", curr_loss
            print "Elapsed time: ", time.time() - start_t
            prev_loss = curr_loss
            curr_loss = 0

            # save model's parameters
            # if (e + 1) % config.save_every == 0:
            #
            #     # print out BLEU scores and file write
            #     # if config.print_bleu:
            #     #     all_gen_cap = np.ndarray((n_examples_val, 16))
            #     #     for i in range(n_iters_val):
            #     #         features_batch = val_captions[i * config.batch_size:(i + 1) * config.batch_size]
            #     #         # feed_dict = {model.level1_model.features: features_batch}
            #     #         gen_cap = generator.beam_search(sess, features_batch)
            #     #         all_gen_cap[i * config.batch_size:(i + 1) * config.batch_size] = gen_cap
            #     #
            #     #     all_decoded = decode_captions(all_gen_cap, model.level1_model.idx_to_word)
            #     #     save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
            #     #     scores = evaluate(data_path='./data', split='val', get_scores=True)
            #     #     write_bleu(scores=scores, path=self.model_path, epoch=e)
            #
            #
            #     saver.save(sess, os.path.join('./model', 'model_level1'), global_step=e + 1)
            #     print "model-%s saved." % (e + 1)




if __name__ == "__main__":
    main()