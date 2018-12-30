import os
import argparse
import json
import numpy as np
import tensorflow as tf
import model.data as data
import model.model_supervised as m
import model.evaluate as eval
import datetime
import json
import sys
import pickle

from math import *
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# sys.setdefaultencoding() does not exist, here!
#reload(sys)  
#sys.setdefaultencoding('UTF8')

os.environ['CUDA_VISIBLE_DEVICES'] = ''
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

home_dir = os.getenv("HOME")

dir(tf.contrib)


def loadGloveModel(gloveFile=None, params=None):
    if gloveFile is None:
        if params.hidden_size == 50:
            gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.50d.txt")
        elif params.hidden_size == 100:
            gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.100d.txt")
        elif params.hidden_size == 200:
            gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.200d.txt")
        elif params.hidden_size == 300:
            gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.300d.txt")
        else:
            print('Invalid dimension [%d] for Glove pretrained embedding matrix!!' %params.hidden_size)
            exit()

    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model

def train(model, dataset, params):
    log_dir = os.path.join(params.model, 'logs')
    model_dir_ir = os.path.join(params.model, 'model_ir')
    model_dir_ppl = os.path.join(params.model, 'model_ppl')
    model_dir_supervised = os.path.join(params.model, 'model_supervised')

    with tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=params.num_cores,
        intra_op_parallelism_threads=params.num_cores,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )) as session:
        avg_loss = tf.placeholder(tf.float32, [], 'loss_ph')
        tf.summary.scalar('loss', avg_loss)

        validation = tf.placeholder(tf.float32, [], 'validation_ph')
        validation_accuracy = tf.placeholder(tf.float32, [], 'validation_acc')
        tf.summary.scalar('validation', validation)
        tf.summary.scalar('validation_accuracy', validation_accuracy)

        summary_writer = tf.summary.FileWriter(log_dir, session.graph)
        summaries = tf.summary.merge_all()
        saver = tf.train.Saver(tf.global_variables())

        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        losses = []

        # This currently streams from disk. You set num_epochs=1 and
        # wrap this call with something like itertools.cycle to keep
        # this data in memory.
        # shuffle: the order of words in the sentence for DocNADE
        if params.bidirectional:
            training_data = dataset.batches_bidirectional('training_docnade', params.batch_size, shuffle=True, multilabel=params.multi_label)
        else:
            training_data = dataset.batches('training_docnade', params.batch_size, shuffle=True, multilabel=params.multi_label)
        # validation_data = dataset.batches('validation_docnade', 50, num_epochs=1, shuffle=True)
        # test_data = dataset.batches('test_docnade', 50, num_epochs=1, shuffle=True)

        best_val_IR = 0.0
        best_val_nll = np.inf
        best_val_ppl = np.inf
        best_val_disc_accuracy = 0.0

        best_test_IR = 0.0
        best_test_nll = np.inf
        best_test_ppl = np.inf
        best_test_disc_accuracy = 0.0
        
        #if params.bidirectional or params.initialize_docnade:
        #    patience = 30
        #else:
        #    patience = params.patience
        
        patience = params.patience

        patience_count = 0
        patience_count_ir = 0
        best_train_nll = np.inf

        training_labels = np.array(
            [[y] for y, _ in dataset.rows('training_docnade', num_epochs=1)]
        )
        validation_labels = np.array(
            [[y] for y, _ in dataset.rows('validation_docnade', num_epochs=1)]
        )
        test_labels = np.array(
            [[y] for y, _ in dataset.rows('test_docnade', num_epochs=1)]
        )

        for step in range(params.num_steps + 1):
            """
            this_train_nll = []
            this_train_ppl_loss = []

            for y, x, seq_lengths in dataset.batches('training_docnade', params.batch_size, num_epochs=1, shuffle=True):
                if params.supervised:
                    _, loss, loss_nll, ppl, disc_loss, disc_accuracy = session.run([model.opt, model.opt_loss, model.loss_ret, model.perplexity,
                                                                        model.disc_loss, model.disc_accuracy], feed_dict={
                        model.x: x,
                        model.y: y,
                        model.seq_lengths: seq_lengths
                    })
                else:
                    _, loss, loss_nll, ppl = session.run([model.opt, model.opt_loss, model.loss_ret, model.perplexity], feed_dict={
                        model.x: x,
                        model.y: y,
                        model.seq_lengths: seq_lengths
                    })
                e_s, h_s, h_t_s, b_s, bias_s, f_s_h_s, l_s, h_r_s, p_a_s = session.run([model.embeddings_shape, model.h_shape, model.h_transpose_shape, model.b_s, model.bias_shape, model.final_state_h_shape, model.logits_shape, model.h_reshaped_shape, model.pre_act_shape], feed_dict={
                    model.x: x,
                    model.y: y,
                    model.seq_lengths: seq_lengths
                })

                indices, embeddings, pre_act, final_h, disc_output, aft_act, last_h = session.run([model.indices, model.embeddings, model.pre_act, model.h, model.disc_output, model.aft_act, model.last_h], feed_dict={
                    model.x: x,
                    model.y: y,
                    model.seq_lengths: seq_lengths
                })

                att, a_m_s, tiled_att_s, w_h_s, w_h, masks, s_w_h, d_h = session.run([model.attentions, model.attentions_mod_shape, model.tiled_attentions_shape, model.weighted_hidden_shape, model.weighted_h, model.masks, model.sum_weighted_hidden, model.disc_h], feed_dict={
                    model.x: x,
                    model.y: y,
                    model.seq_lengths: seq_lengths
                })
                this_train_nll.append(loss_nll)
                this_train_ppl_loss.append(loss)

            total_train_nll = np.mean(this_train_nll)
            total_train_ppl_loss = np.mean(this_train_ppl_loss)
            total_train_ppl = np.exp(total_train_ppl_loss)
            
            # Early stopping
            if total_train_nll < best_train_nll:
                best_train_nll = total_train_nll
                patience_count = 0
            else:
                patience_count += 1

            if patience_count > patience:
                print("Early stopping criterion satisfied.")
                break
            
            if params.supervised:
                losses.append(total_train_nll + disc_loss)
            else:
                losses.append(total_train_nll)

            if (step % params.log_every == 0):
                print('{}: {:.6f}'.format(step, total_train_nll))
                
            if params.supervised:
                print("accuracy: ", disc_accuracy)
            """
            this_loss = -1.
            if params.bidirectional:
                y, x, x_bw, seq_lengths = next(training_data)

                if params.supervised:
                    _, loss_normed, loss_unnormed, loss_normed_bw, loss_unnormed_bw, disc_loss, disc_accuracy = session.run([model.opt, model.loss_normed, model.loss_unnormed,
                                                                                                        model.loss_normed_bw, model.loss_unnormed_bw,
                                                                                                        model.disc_loss, model.disc_accuracy], feed_dict={
                        model.x: x,
                        model.x_bw: x_bw,
                        model.y: y,
                        model.seq_lengths: seq_lengths
                    })
                    this_loss = 0.5 * (loss_unnormed + loss_unnormed_bw) + disc_loss
                    losses.append(this_loss)
                else:
                    _, loss_normed, loss_unnormed, loss_normed_bw, loss_unnormed_bw = session.run([model.opt, model.loss_normed, model.loss_unnormed,
                                                                                model.loss_normed_bw, model.loss_unnormed_bw], feed_dict={
                        model.x: x,
                        model.x_bw: x_bw,
                        model.y: y,
                        model.seq_lengths: seq_lengths
                    })
                    this_loss = 0.5 * (loss_unnormed + loss_unnormed_bw)
                    losses.append(this_loss)
            else:
                y, x, seq_lengths = next(training_data)
            
                if params.supervised:
                    _, loss, loss_unnormed, disc_loss, disc_accuracy = session.run([model.opt, model.loss_normed, model.loss_unnormed,
                                                                            model.disc_loss, model.disc_accuracy], feed_dict={
                        model.x: x,
                        model.y: y,
                        model.seq_lengths: seq_lengths
                    })
                    this_loss = loss + disc_loss
                    losses.append(this_loss)
                else:
                    _, loss, loss_unnormed = session.run([model.opt, model.loss_normed, model.loss_unnormed], feed_dict={
                        model.x: x,
                        model.y: y,
                        model.seq_lengths: seq_lengths
                    })
                    this_loss = loss
                    losses.append(this_loss)

            if (step % params.log_every == 0):
                print('{}: {:.6f}'.format(step, this_loss))
                
            #if params.supervised:
            #    print("accuracy: ", disc_accuracy)

            if step and (step % params.validation_ppl_freq) == 0:
                this_val_nll = []
                this_val_loss_normed = []
                # val_loss_unnormed is NLL
                this_val_nll_bw = []
                this_val_loss_normed_bw = []

                this_val_disc_accuracy = []

                if params.bidirectional:
                    for val_y, val_x, val_x_bw, val_seq_lengths in dataset.batches_bidirectional('validation_docnade', params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                        if params.supervised:
                            val_loss_normed, val_loss_unnormed, val_loss_normed_bw, \
                            val_loss_unnormed_bw, val_disc_loss, val_disc_accuracy = session.run([model.loss_normed, model.loss_unnormed,
                                                                                                model.loss_normed_bw, model.loss_unnormed_bw, 
                                                                                                model.disc_loss, model.disc_accuracy], feed_dict={
                                model.x: val_x,
                                model.x_bw: val_x_bw,
                                model.y: val_y,
                                model.seq_lengths: val_seq_lengths
                            })
                            this_val_disc_accuracy.append(val_disc_accuracy)
                        else:
                            val_loss_normed, val_loss_unnormed, \
                            val_loss_normed_bw, val_loss_unnormed_bw = session.run([model.loss_normed, model.loss_unnormed, 
                                                                                    model.loss_normed_bw, model.loss_unnormed_bw], feed_dict={
                                model.x: val_x,
                                model.x_bw: val_x_bw,
                                model.y: val_y,
                                model.seq_lengths: val_seq_lengths
                            })
                        this_val_nll.append(val_loss_unnormed)
                        this_val_loss_normed.append(val_loss_normed)
                        this_val_nll_bw.append(val_loss_unnormed_bw)
                        this_val_loss_normed_bw.append(val_loss_normed_bw)
                else:
                    for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                        if params.supervised:
                            val_loss_normed, val_loss_unnormed, val_disc_loss, val_disc_accuracy = session.run([model.loss_normed, model.loss_unnormed,
                                                                                                                    model.disc_loss, model.disc_accuracy], feed_dict={
                                model.x: val_x,
                                model.y: val_y,
                                model.seq_lengths: val_seq_lengths
                            })
                            this_val_disc_accuracy.append(val_disc_accuracy)
                        else:
                            val_loss_normed, val_loss_unnormed = session.run([model.loss_normed, model.loss_unnormed], feed_dict={
                                model.x: val_x,
                                model.y: val_y,
                                model.seq_lengths: val_seq_lengths
                            })
                        this_val_nll.append(val_loss_unnormed)
                        this_val_loss_normed.append(val_loss_normed)
                
                if params.bidirectional:
                    total_val_nll = 0.5 * (np.mean(this_val_nll) + np.mean(this_val_nll_bw))
                    total_val_ppl = 0.5 * (np.exp(np.mean(this_val_loss_normed)) + np.exp(np.mean(this_val_loss_normed_bw)))
                else:
                    total_val_nll = np.mean(this_val_nll)
                    total_val_ppl = np.exp(np.mean(this_val_loss_normed))

                if total_val_ppl < best_val_ppl:
                    best_val_ppl = total_val_ppl
                    print('saving: {}'.format(model_dir_ppl))
                    saver.save(session, model_dir_ppl + '/model_ppl', global_step=1)

                # Early stopping
                if total_val_nll < best_val_nll:
                    best_val_nll = total_val_nll
                    patience_count = 0
                else:
                    patience_count += 1

                

                print('This val PPL: {:.3f} (best val PPL: {:.3f},  best val loss: {:.3f})'.format(
                    total_val_ppl,
                    best_val_ppl or 0.0,
                    best_val_nll
                ))

                # logging information
                with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                    f.write("Step: %i,    val PPL: %s,     best val PPL: %s,    best val loss: %s\n" % 
                            (step, total_val_ppl, best_val_ppl, best_val_nll))

                if patience_count > patience:
                    print("Early stopping criterion satisfied.")
                    break
            
            #if patience_count_ir > 10 and params.deep:
            #    params.validation_ir_freq = 1000000
            
            if step >= 1 and step % params.validation_ir_freq == 0:
                if params.supervised:
                    total_val_disc_accuracy = np.mean(this_val_disc_accuracy)

                    if total_val_disc_accuracy > best_val_disc_accuracy:
                        best_val_disc_accuracy = total_val_disc_accuracy
                        print('saving: {}'.format(model_dir_supervised))
                        saver.save(session, model_dir_supervised + '/model_supervised', global_step=1)

                    print('This val accuracy: {:.3f} (best val accuracy: {:.3f})'.format(
                        total_val_disc_accuracy,
                        best_val_disc_accuracy or 0.0
                    ))

                    # logging information
                    with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                        f.write("Step: %i,    This val accuracy: %s,    best val accuracy: %s\n" % 
                                (step, total_val_disc_accuracy, best_val_disc_accuracy))

                    if params.bidirectional:
                        summary, = session.run([summaries], feed_dict={
                            model.x: x,
                            model.x_bw: x_bw,
                            model.y: y,
                            model.seq_lengths: seq_lengths,
                            validation: 0.0,
                            validation_accuracy: total_val_disc_accuracy,
                            avg_loss: np.average(losses)
                        })
                    else:
                        summary, = session.run([summaries], feed_dict={
                            model.x: x,
                            model.y: y,
                            model.seq_lengths: seq_lengths,
                            validation: 0.0,
                            validation_accuracy: total_val_disc_accuracy,
                            avg_loss: np.average(losses)
                        })
                    summary_writer.add_summary(summary, step)
                    summary_writer.flush()
                    losses = []
                else:
                    if params.bidirectional:
                        validation_vectors = m.vectors_bidirectional(
                            model,
                            dataset.batches_bidirectional(
                                'validation_docnade',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            session,
                            params.combination_type
                        )

                        training_vectors = m.vectors_bidirectional(
                            model,
                            dataset.batches_bidirectional(
                                'training_docnade',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            session,
                            params.combination_type
                        )
                    else:
                        validation_vectors = m.vectors(
                            model,
                            dataset.batches(
                                'validation_docnade',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            session
                        )

                        training_vectors = m.vectors(
                            model,
                            dataset.batches(
                                'training_docnade',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            session
                        )

                    val = eval.evaluate(
                        training_vectors,
                        validation_vectors,
                        training_labels,
                        validation_labels,
                        recall=[0.02],
                        num_classes=params.num_classes,
                        multi_label=params.multi_label
                    )[0]

                    if val > best_val_IR:
                        best_val_IR = val
                        print('saving: {}'.format(model_dir_ir))
                        saver.save(session, model_dir_ir + '/model_ir', global_step=1)
                        patience_count_ir = 0
                    else:
                        patience_count_ir += 1
                    
                    print('This val IR: {:.3f} (best val IR: {:.3f})'.format(
                        val,
                        best_val_IR or 0.0
                    ))

                    # logging information
                    with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                        f.write("Step: %i,    val IR: %s,    best val IR: %s\n" % 
                                (step, val, best_val_IR))

                    if params.bidirectional:
                        summary, = session.run([summaries], feed_dict={
                            model.x: x,
                            model.x_bw: x_bw,
                            model.y: y,
                            model.seq_lengths: seq_lengths,
                            validation: val,
                            validation_accuracy: 0.0,
                            avg_loss: np.average(losses)
                        })
                    else:
                        summary, = session.run([summaries], feed_dict={
                            model.x: x,
                            model.y: y,
                            model.seq_lengths: seq_lengths,
                            validation: val,
                            validation_accuracy: 0.0,
                            avg_loss: np.average(losses)
                        })
                    summary_writer.add_summary(summary, step)
                    summary_writer.flush()
                    losses = []

                if patience_count_ir > patience:
                    print("Early stopping criterion satisfied.")
                    break
            
            if step and (step % params.test_ppl_freq) == 0:
                this_test_nll = []
                this_test_loss_normed = []
                this_test_nll_bw = []
                this_test_loss_normed_bw = []
                this_test_disc_accuracy = []

                if params.bidirectional:
                    for test_y, test_x, test_x_bw, test_seq_lengths in dataset.batches_bidirectional('test_docnade', params.test_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                        if params.supervised:
                            test_loss_normed, test_loss_unnormed, test_loss_normed_bw, \
                            test_loss_unnormed_bw, test_disc_loss, test_disc_accuracy = session.run([model.loss_normed, model.loss_unnormed, model.loss_normed_bw, 
                                                                                                    model.loss_unnormed_bw, model.disc_loss, model.disc_accuracy], feed_dict={
                                model.x: test_x,
                                model.x_bw: test_x_bw,
                                model.y: test_y,
                                model.seq_lengths: test_seq_lengths
                            })
                            this_test_disc_accuracy.append(test_disc_accuracy)
                        else:
                            test_loss_normed, test_loss_unnormed, \
                             test_loss_normed_bw, test_loss_unnormed_bw = session.run([model.loss_normed, model.loss_unnormed, 
                                                                                        model.loss_normed_bw, model.loss_unnormed_bw], feed_dict={
                                model.x: test_x,
                                model.x_bw: test_x_bw,
                                model.y: test_y,
                                model.seq_lengths: test_seq_lengths
                            })
                        this_test_nll.append(test_loss_unnormed)
                        this_test_loss_normed.append(test_loss_normed)
                        this_test_nll_bw.append(test_loss_unnormed_bw)
                        this_test_loss_normed_bw.append(test_loss_normed_bw)
                else:
                    for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params.test_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                        if params.supervised:
                            test_loss_normed, test_loss_unnormed, test_disc_loss, test_disc_accuracy = session.run([model.loss_normed, model.loss_unnormed, 
                                                                                                                        model.disc_loss, model.disc_accuracy], feed_dict={
                                model.x: test_x,
                                model.y: test_y,
                                model.seq_lengths: test_seq_lengths
                            })
                            this_test_disc_accuracy.append(test_disc_accuracy)
                        else:
                            test_loss_normed, test_loss_unnormed = session.run([model.loss_normed, model.loss_unnormed], feed_dict={
                                model.x: test_x,
                                model.y: test_y,
                                model.seq_lengths: test_seq_lengths
                            })
                        this_test_nll.append(test_loss_unnormed)
                        this_test_loss_normed.append(test_loss_normed)

                if params.bidirectional:
                    total_test_nll = 0.5 * (np.mean(this_test_nll) + np.mean(this_test_nll_bw))
                    total_test_ppl = 0.5 * (np.exp(np.mean(this_test_loss_normed)) + np.exp(np.mean(this_test_loss_normed_bw)))
                else:
                    total_test_nll = np.mean(this_test_nll)
                    total_test_ppl = np.exp(np.mean(this_test_loss_normed))

                if total_test_ppl < best_test_ppl:
                    best_test_ppl = total_test_ppl

                if total_test_nll < best_test_nll:
                    best_test_nll = total_test_nll

                print('This test PPL: {:.3f} (best test PPL: {:.3f},  best test loss: {:.3f})'.format(
                    total_test_ppl,
                    best_test_ppl or 0.0,
                    best_test_nll
                ))

                # logging information
                with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                    f.write("Step: %i,    test PPL: %s,    best test PPL: %s,    best test loss: %s\n" % 
                            (step, total_test_ppl, best_test_ppl, best_test_nll))

            
            if step >= 1 and (step % params.test_ir_freq) == 0:
                if params.supervised:
                    total_test_disc_accuracy = np.mean(this_test_disc_accuracy)

                    if total_test_disc_accuracy > best_test_disc_accuracy:
                        best_test_disc_accuracy = total_test_disc_accuracy

                    print('This test accuracy: {:.3f} (best test accuracy: {:.3f})'.format(
                        total_test_disc_accuracy,
                        best_test_disc_accuracy or 0.0
                    ))

                    # logging information
                    with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                        f.write("Step: %i,    This test accuracy: %s,    best test accuracy: %s\n" % 
                                (step, total_test_disc_accuracy, best_test_disc_accuracy))
                else:
                    if params.bidirectional:
                        test_vectors = m.vectors_bidirectional(
                            model,
                            dataset.batches_bidirectional(
                                'test_docnade',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            session,
                            params.combination_type
                        )

                        training_vectors = m.vectors_bidirectional(
                            model,
                            dataset.batches_bidirectional(
                                'training_docnade',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            session,
                            params.combination_type
                        )
                    else:
                        test_vectors = m.vectors(
                            model,
                            dataset.batches(
                                'test_docnade',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            session
                        )

                        training_vectors = m.vectors(
                            model,
                            dataset.batches(
                                'training_docnade',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            session
                        )

                    test = eval.evaluate(
                        training_vectors,
                        test_vectors,
                        training_labels,
                        test_labels,
                        recall=[0.02],
                        num_classes=params.num_classes,
                        multi_label=params.multi_label
                    )[0]

                    if test > best_test_IR:
                        best_test_IR = test
                    
                    print('This test IR: {:.3f} (best test IR: {:.3f})'.format(
                        test,
                        best_test_IR or 0.0
                    ))

                    # logging information
                    with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                        f.write("Step: %i,    test IR: %s,    best test IR: %s\n" % 
                            (step, test, best_test_IR))


def get_vectors_from_matrix(matrix, batches):
    # matrix: embedding matrix of shape = [vocab_size X embedding_size]
    vecs = []
    for _, x, seq_length in batches:
        temp_vec = np.zeros((matrix.shape[1]), dtype=np.float32)
        indices = x[0, :seq_length[0]]
        for index in indices:
            temp_vec += matrix[index, :]
        vecs.append(temp_vec)
    return np.array(vecs)

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)

def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)

def reload_evaluation(model_ir, model_ppl, dataset, params):
    return
	
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    args.reload = str2bool(args.reload)
    args.supervised = str2bool(args.supervised)
    args.initialize_docnade = str2bool(args.initialize_docnade)
    args.bidirectional = str2bool(args.bidirectional)
    args.projection = str2bool(args.projection)
    args.deep = str2bool(args.deep)
    args.multi_label = str2bool(args.multi_label)
    args.shuffle_reload = str2bool(args.shuffle_reload)

    x = tf.placeholder(tf.int32, shape=(None, None), name='x')
    x_bw = tf.placeholder(tf.int32, shape=(None, None), name='x_bw')
    if args.multi_label:
        y = tf.placeholder(tf.string, shape=(None), name='y')
    else:
        y = tf.placeholder(tf.int32, shape=(None), name='y')
    seq_lengths = tf.placeholder(tf.int32, shape=(None), name='seq_lengths')

    if args.reload:
	# to do
       return 
    else:
        now = datetime.datetime.now()

        if args.bidirectional:
            args.model += "_iDocNADE"
        else:
            args.model += "_DocNADE"

        if args.supervised:
            args.model += "_supervised"

        if args.initialize_docnade:
            args.model += "_embprior_lambda_" + str(args.lambda_hidden_lstm)
        
        args.model +=  "_act_" + str(args.activation) + "_hidden_" + str(args.hidden_size) + "_vocab_" + str(args.vocab_size) \
                        + "_lr_" + str(args.learning_rate) + "_proj_" + str(args.projection) + "_deep_" + str(args.deep) \
                        + "_" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)
        
        if not os.path.isdir(args.model):
            os.mkdir(args.model)

        docnade_vocab = args.docnadeVocab

        #if args.bidirectional or args.initialize_docnade:
        #    args.patience = 30

        with open(os.path.join(args.model, 'params.json'), 'w') as f:
            f.write(json.dumps(vars(args)))

        dataset = data.Dataset(args.dataset)

        if args.initialize_docnade:
            glove_embeddings = loadGloveModel(params=args)

        #training_batches = dataset.batches('training', 1000, num_epochs=1, shuffle=True)
        #for y, x, seq_lengths in training_batches:
        #    print("max length: ", x.shape[1])


        with open(docnade_vocab, 'r') as f:
            vocab_docnade = [w.strip() for w in f.readlines()]

        docnade_embedding_matrix = None
        if args.initialize_docnade:
            missing_words = 0
            docnade_embedding_matrix = np.zeros((len(vocab_docnade), args.hidden_size), dtype=np.float32)
            for i, word in enumerate(vocab_docnade):
                if str(word).lower() in glove_embeddings.keys():
                    if len(glove_embeddings[str(word).lower()]) == 0:
                        docnade_embedding_matrix[i, :] = np.zeros((args.hidden_size), dtype=np.float32)
                        missing_words += 1
                    else:
                        docnade_embedding_matrix[i, :] = np.array(glove_embeddings[str(word).lower()], dtype=np.float32)
                else:
                    docnade_embedding_matrix[i, :] = np.zeros((args.hidden_size), dtype=np.float32)
                    missing_words += 1

            docnade_embedding_matrix = tf.convert_to_tensor(docnade_embedding_matrix)
            print("Total missing words:%d out of %d" %(missing_words, len(vocab_docnade)))

        docnade_pretrained_matrix = None
        if args.pretrained_embeddings_path:
            with open(args.pretrained_embeddings_path, "rb") as f:
                docnade_pretrained_matrix = pickle.load(f)
            print("pretrained embeddings loaded.")
        
        if args.bidirectional:
            model = m.iDocNADE(x, x_bw, y, seq_lengths, args, W_initializer=docnade_embedding_matrix, lambda_hidden_lstm=args.lambda_hidden_lstm, W_pretrained=docnade_pretrained_matrix)
            print("iDocNADE created")
        else:
            model = m.DocNADE(x, y, seq_lengths, args, W_initializer=docnade_embedding_matrix, lambda_hidden_lstm=args.lambda_hidden_lstm, W_pretrained=docnade_pretrained_matrix)
            print("DocNADE created")
        
        train(model, dataset, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='path to model output directory')
    parser.add_argument('--dataset', type=str, required=True,
                        help='path to the input dataset')
    parser.add_argument('--vocab-size', type=int, default=2000,
                        help='the vocab size')
    parser.add_argument('--hidden-size', type=int, default=50,
                        help='size of the hidden layer')
    parser.add_argument('--activation', type=str, default='tanh',
                        help='which activation to use: sigmoid|tanh')
    parser.add_argument('--learning-rate', type=float, default=0.0004,
                        help='initial learning rate')
    parser.add_argument('--num-steps', type=int, default=50000,
                        help='the number of steps to train for')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='softmax samples (default: full softmax)')
    parser.add_argument('--num-cores', type=int, default=2,
                        help='the number of CPU cores to use')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--validation-ppl-freq', type=int, default=500,
                        help='print loss after this many steps')

    parser.add_argument('--num-classes', type=int, default=-1,
                        help='number of classes')
    parser.add_argument('--supervised', type=str, default="False",
                        help='whether to use supervised model or not')
    #parser.add_argument('--hidden-sizes', nargs='+', type=int,
    #                    help='sizes of the hidden layers')

    parser.add_argument('--initialize-docnade', type=str, default="False",
                        help='whether to embedding matrix of docnade')
    parser.add_argument('--docnadeVocab', type=str, default="False",
                        help='path to vocabulary file used by DocNADE')
    parser.add_argument('--test-ppl-freq', type=int, default=100,
                        help='print and log test PPL after this many steps')
    parser.add_argument('--test-ir-freq', type=int, default=100,
                        help='print and log test IR after this many steps')
    parser.add_argument('--patience', type=int, default=10,
                        help='print and log test IR after this many steps')
    parser.add_argument('--validation-bs', type=int, default=64,
                        help='the batch size for validation evaluation')
    parser.add_argument('--test-bs', type=int, default=64,
                        help='the batch size for test evaluation')
    parser.add_argument('--validation-ir-freq', type=int, default=500,
                        help='print loss after this many steps')
    parser.add_argument('--bidirectional', type=str, default="False",
                        help='whether to use bidirectional DocNADE model or not')
    parser.add_argument('--combination-type', type=str, default="concat",
                        help='combination type for bidirectional docnade')
    parser.add_argument('--generative-loss-weight', type=float, default=10.0,
                        help='weight for generative loss in total loss')
    parser.add_argument('--projection', type=str, default="False",
                        help='whether to project prior embeddings or not')
    parser.add_argument('--reload', type=str, default="False",
                        help='whether to reload model or not')
    parser.add_argument('--reload-model-dir', type=str,
                        help='path for model to be reloaded')
    parser.add_argument('--model-type', type=str,
                        help='type of model to be reloaded')
    parser.add_argument('--deep-hidden-sizes', nargs='+', type=int,
                        help='sizes of the hidden layers')
    parser.add_argument('--deep', type=str, default="False",
                        help='whether to maked model deep or not')
    parser.add_argument('--multi-label', type=str, default="False",
                        help='whether dataset is multi-label or not')
    parser.add_argument('--shuffle-reload', type=str, default="True",
                        help='whether dataset is shuffled or not')
    parser.add_argument('--trainfile', type=str, required=True,
                        help='path to train text file')
    parser.add_argument('--valfile', type=str, required=True,
                        help='path to validation text file')
    parser.add_argument('--testfile', type=str, required=True,
                        help='path to test text file')
    parser.add_argument('--lambda-hidden-lstm', type=float, default=0.0,
                        help='combination weight for prior embeddings into docnade')
    parser.add_argument('--pretrained-embeddings-path', type=str, default="",
                        help='path for pretrained embeddings')


    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
