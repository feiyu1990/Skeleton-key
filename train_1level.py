import tensorflow as tf
from utils import *
from model_stem import Level1Model
from beam_search import CaptionGenerator
import config

def main():
    model = Level1Model(config, mode='training', train_resnet=config.train_resnet)

    data = load_coco_data(data_path='./data', split='testtest')
    val_data = load_coco_data(data_path='./data', split='testtest')

    # update_rule = 'adam'
    optimizer = tf.train.AdamOptimizer
    log_path = './model/level1_'
    pretrained_model = None

    generator = CaptionGenerator(model, model.level1_word2ix, None,
                                 beam_size_1level=3, beam_size_2level=None,
                                 encourage_1level=0.0, encourage_2level=None,
                                 level2=False)
    loss = model.build()
    # with tf.Session() as sess:
    #     loss = model.build()
    #     tf.global_variables_initializer().run()
    #     # saver = tf.train.Saver()
    #     # saver.restore(sess, './model/pretrained_model-0')
    #     features_batch, image_files = sample_coco_minibatch(data, 1)
    #
    #     # generator.beam_search(sess, features_batch)

    n_examples = data['captions'].shape[0]
    n_iters_per_epoch = int(np.ceil(float(n_examples) / config.batch_size))
    features = data['features'][:]
    captions = data['captions'][:]
    image_idxs = data['img_idxs'][:]
    val_features = val_data['features'][:]
    n_iters_val = int(np.ceil(float(val_features.shape[0]) / config.batch_size))

    with tf.name_scope('optimizer'):
        optimizer = optimizer(learning_rate=4e-6)
        optim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='level1')
        level1_grads = tf.gradients(loss, optim_vars)
        # for i, j in zip(level1_grads, optim_vars):
        #     print i, j
        grads_and_vars = list(zip(level1_grads, optim_vars))
        # todo: here check the batch-norm moving average/var
        # print '************************'
        if config.train_resnet:
            optim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet')
            resnet_grads = tf.gradients(model.resnet.features, optim_vars)
            resnet_pairs = [(i, j) for i, j in zip(resnet_grads, optim_vars) if i != None]
            grads_and_vars.extend(resnet_pairs)
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

    # summary op
    print '************************'
    tf.summary.scalar('batch_loss', loss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    for grad, var in grads_and_vars:
        # print grad, var
        tf.summary.histogram(var.op.name + '/gradient', grad)

    summary_op = tf.summary.merge_all()

    print "The number of epoch: %d" % config.n_epochs
    print "Data size: %d" % n_examples
    print "Batch size: %d" % config.batch_size
    print "Iterations per epoch: %d" % n_iters_per_epoch

    config_ = tf.ConfigProto(allow_soft_placement=True)
    # config_.gpu_options.per_process_gpu_memory_fraction=0.9
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

        for e in range(config.n_epochs):
            rand_idxs = np.random.permutation(n_examples)
            # print captions.shape, image_idxs.shape
            # print captions
            captions = captions[rand_idxs]
            # print rand_idxs
            # print captions
            image_idxs = image_idxs[rand_idxs]

            for i in range(n_iters_per_epoch):
                captions_batch = captions[i * config.batch_size:(i + 1) * config.batch_size]
                image_idxs_batch = image_idxs[i * config.batch_size:(i + 1) * config.batch_size]
                img_batch = features[image_idxs_batch]
                print img_batch.shape
                img_feature = sess.run(model.resnet.features, {model.resnet.images: img_batch})
                print img_feature.shape
                feed_dict = {model.level1_model.features: img_feature, model.level1_model.captions: captions_batch,
                             model.resnet.images: img_batch}
                _, l = sess.run([train_op, loss], feed_dict)
                curr_loss += l

                # write summary for tensorboard visualization
                if i % 10 == 0:
                    summary = sess.run(summary_op, feed_dict)
                    summary_writer.add_summary(summary, e * n_iters_per_epoch + i)

                if (i + 1) % config.print_every == 0:
                    print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" % (e + 1, i + 1, l)
                    ground_truths = captions[image_idxs == image_idxs_batch[0]]
                    decoded = decode_captions(ground_truths, model.level1_model.idx_to_word)
                    for j, gt in enumerate(decoded):
                        print "Ground truth %d: %s" % (j + 1, gt)
                    print img_batch.shape
                    generator.beam_search(sess, img_batch[0,:,:,:])
                    print "Generated caption: %s\n" % generator

            print "Previous epoch loss: ", prev_loss
            print "Current epoch loss: ", curr_loss
            print "Elapsed time: ", time.time() - start_t
            prev_loss = curr_loss
            curr_loss = 0

            # # print out BLEU scores and file write
            # if config.print_bleu:
            #     all_gen_cap = np.ndarray((val_features.shape[0], 20))
            #     for i in range(n_iters_val):
            #         features_batch = val_features[i * config.batch_size:(i + 1) * config.batch_size]
            #         feed_dict = {model.level1_model.features: features_batch}
            #         gen_cap = generator.beam_search(sess, features_batch)
            #         all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gen_cap
            #
            #     all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
            #     save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
            #     scores = evaluate(data_path='./data', split='val', get_scores=True)
            #     write_bleu(scores=scores, path=self.model_path, epoch=e)
            #
            # # save model's parameters
            # if (e + 1) % self.save_every == 0:
            #     saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e + 1)
            #     print "model-%s saved." % (e + 1)




if __name__ == "__main__":
    main()