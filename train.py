import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import signal
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import (
    BinaryCrossentropy,
    MeanAbsoluteError,
    MeanSquaredError,
)
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

import model


class WdsrTrainer:
    def __init__(
        self,
        model,
        loss=MeanAbsoluteError(),
        learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4]),
        checkpoint_dir="./ckpt/edsr",
        nbit=16,
        fn_kernel=None,
    ):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            psnr=tf.Variable(-1.0),
            optimizer=Adam(learning_rate),
            model=model,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint, directory=checkpoint_dir, max_to_keep=3
        )

        self.restore()
        if fn_kernel is not None:
            self.kernel = np.load(fn_kernel)
        else:
            self.kernel = None

    @property
    def model(self):
        return self.checkpoint.model

    def train(
        self,
        train_dataset,
        valid_dataset,
        steps=300000,
        evaluate_every=1000,
        save_best_only=True,
        nbit=16,
        fnoutweights=None,
    ):
        # setup logs
        if fnoutweights is None:
            fnoutweights = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_logdir = os.path.join("logs", "train", fnoutweights)
        val_logdir = os.path.join("logs", "val", fnoutweights)
        img_logdir = os.path.join("logs", "img", fnoutweights)
        train_summary_writer = tf.summary.create_file_writer(train_logdir)
        val_summary_writer = tf.summary.create_file_writer(val_logdir)
        img_summary_writer = tf.summary.create_file_writer(img_logdir)
        print("Writing logs to %s" % train_logdir)
        tb_callback = tf.keras.callbacks.TensorBoard(train_logdir, histogram_freq=1)
        tb_callback.set_model(self.model)

        # setup metrics
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()
        print("Training begins @ %s" % self.now)

        print(steps - ckpt.step.numpy())
        print(train_dataset.take(steps - ckpt.step.numpy()))
        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()
            loss = self.train_step(lr, hr)
            loss_mean(loss)
            # if there's any issues, make sure it logs it out early
            if step < 20:
                loss_value = loss_mean.result()
                duration = time.perf_counter() - self.now
                print(f"{step}/{steps}: loss = {loss_value.numpy():.3f}")
                self.now = time.perf_counter()

                with train_summary_writer.as_default():
                    tf.summary.scalar("loss", loss_mean.result(), step=step)

            if step % evaluate_every == 0 or step == 20:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                psnr_value = self.evaluate(valid_dataset, nbit=nbit, show_image=True)

                # update the logs
                with train_summary_writer.as_default():
                    tf.summary.scalar("loss", loss_value, step=step)
                duration = time.perf_counter() - self.now
                print(
                    f"{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)"
                )

                for tf_var in self.model.trainable_weights:
                    # plot a histogram of the tensor values
                    plt.hist(tf_var.numpy().flatten(), bins=100)
                    plt.title("histogram of %s @%s" % (tf_var.name, str(step)))
                plt.show()

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

    def kernel_loss(self, sr, lr):
        lr_estimate = signal.fftconvolve(sr.numpy(), self.kernel, mode="same")
        print(lr.shape, lr_estimate[2::4, 2::4].shape)
        raise Exception("kernel loss called")
        exit()

    @tf.function
    def train_step(self, lr, hr, gg=1.0):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(sr, hr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables)
        )

        return loss_value

    def evaluate(self, dataset, nbit=16, show_image=False):
        return model.evaluate(
            self.checkpoint.model,
            dataset,
            nbit=nbit,
            show_image=show_image,
            loss_name=self.loss.__name__,
        )

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f"Model restored from checkpoint at step {self.checkpoint.step.numpy()}."
            )
