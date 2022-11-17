import datetime
import os

from scipy import signal
import numpy as np
import time
import tensorflow as tf

from model import evaluate

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from reconstruct import plot_reconstruction


class Trainer:
    def __init__(
        self,
        model,
        loss,
        learning_rate,
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
        steps,
        evaluate_every=1000,
        save_best_only=False,
        nbit=16,
        fnoutweights=None,
    ):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()
        self.begin = time.perf_counter()
        print("Training begins @ %s" % self.now)
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

        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()
            #            lr = tf.cast(lr, tf.float32)

            #            lr = tf.image.adjust_gamma(lr, 0.5)
            #            print(tf.math.reduce_max(lr),tf.math.reduce_min(lr))
            loss = -1
            if step % evaluate_every == 0 or step == 2:
                loss = self.train_step(lr, hr, show_parts=True)
            else:
                loss = self.train_step(lr, hr)
            # print('wtf loss', loss)
            loss_mean(loss)

            if step < 20:
                loss_value = loss_mean.result()
                duration = time.perf_counter() - self.now
                print(f"{step}/{steps}: loss = {loss_value.numpy():.3f}")
                self.now = time.perf_counter()

                with train_summary_writer.as_default():
                    tf.summary.scalar("loss", loss_mean.result(), step=step)

            if step % evaluate_every == 0 or step == 2:

                with train_summary_writer.as_default():
                    tf.summary.scalar("loss", loss_mean.result(), step=step)

                loss_value = loss_mean.result()
                loss_mean.reset_states()
                # Compute PSNR on validation dataset
                (psnr_value, example_img) = self.evaluate(
                    valid_dataset, nbit=nbit, show_image=True
                )
                # tf.print(example_img)
                # tf.print(example_img['lr'])
                # tf.print('lr shape', example_img['lr'].shape)
                # tf.print('hr shape', example_img['hr'].shape)
                # tf.print(   'sr shape', example_img['sr'].shape)
                # tf.print('uq shape', example_img['uq'].shape)
                lr = example_img["lr"]
                lr = tf.reshape(lr, (lr.shape[1], lr.shape[2], 1))
                hr = example_img["hr"]
                hr = tf.reshape(hr, (hr.shape[1], hr.shape[2]))
                sr = example_img["sr"]
                sr = tf.reshape(sr, (sr.shape[1], sr.shape[2]))
                uq = example_img["uq"]
                uq = tf.reshape(uq, (uq.shape[1], uq.shape[2]))
                plot_reconstruction(
                    lr, sr, hr,
                    mc_data=uq,
                    vm=1,
                    nsub=4,
                    regular_image=False,
                )
                with val_summary_writer.as_default():
                    tf.summary.scalar("psnr", psnr_value, step=step)
                duration = time.perf_counter() - self.now
                print(
                    f"{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)"
                )

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue
                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()
        print("Done training @ %s" % self.now)

    def kernel_loss(self, sr, lr):
        lr_estimate = signal.fftconvolve(sr.numpy(), self.kernel, mode="same")
        # lr_estimate = tf.nn.conv2d(sr, kernel, strides=[1, 1, 1, 1], padding='VALID')

        print(lr.shape, lr_estimate[2::4, 2::4].shape)
        raise Exception

    #     return self.loss(lr, lr_estimate)

    # @tf.function
    # def train_step(self, lr, hr):
    #     with tf.GradientTape() as tape:
    #         lr = tf.cast(lr, tf.float32)
    #         hr = tf.cast(hr, tf.float32)
    #         sr = self.checkpoint.model(lr, training=True)
    #         loss_value = self.loss(hr, sr)

    #     gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
    #     self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

    #     return loss_value

    @tf.function
    def train_step(self, lr, hr, gg=1.0, show_parts = False):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            #            lr = tf.image.adjust_gamma(lr,0.9)
            #            hr = tf.image.adjust_gamma(hr,0.9)
            sr = self.checkpoint.model(lr, training=True)
            # tf.print('sr_shape during trainnig', sr.shape)
            #            sr_ = sr - tf.reduce_min(sr)
            #            hr_ = hr - tf.reduce_min(hr)
            loss_value = self.loss(sr, hr, show_parts=show_parts)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables)
        )

        return loss_value

    def evaluate(self, dataset, nbit=16, show_image=False):
        # print('step in evaluate')
        return evaluate(
            self.checkpoint.model, dataset, nbit=nbit, show_image=show_image
        )

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f"Model restored from checkpoint at step {self.checkpoint.step.numpy()}."
            )


class EdsrTrainer(Trainer):
    def __init__(
        self,
        model,
        checkpoint_dir,
        learning_rate=PiecewiseConstantDecay(
            boundaries=[50000, 200000], values=[1e-4, 1e-5, 5e-6]
        ),
    ):
        super().__init__(
            model,
            loss=MeanAbsoluteError(),
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
        )

    def train(
        self,
        train_dataset,
        valid_dataset,
        steps=300000,
        evaluate_every=1000,
        save_best_only=True,
    ):
        super().train(
            train_dataset, valid_dataset, steps, evaluate_every, save_best_only
        )


class WdsrTrainer(Trainer):
    def __init__(
        self,
        model,
        checkpoint_dir,
        learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4]),
        nbit=16,
        fn_kernel=None,
        loss=MeanAbsoluteError(),
    ):
        super().__init__(
            model,
            loss=loss,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
            fn_kernel=fn_kernel,
        )

    def train(
        self,
        train_dataset,
        valid_dataset,
        steps=300000,
        evaluate_every=1000,
        save_best_only=True,
        fnoutweights=None,
    ):
        super().train(
            train_dataset,
            valid_dataset,
            steps,
            evaluate_every,
            save_best_only,
            fnoutweights=fnoutweights,
        )


class SrganGeneratorTrainer(Trainer):
    def __init__(self, model, checkpoint_dir, learning_rate=1e-4):
        super().__init__(
            model,
            loss=MeanSquaredError(),
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
        )

    def train(
        self,
        train_dataset,
        valid_dataset,
        steps=1000000,
        evaluate_every=1000,
        save_best_only=True,
    ):
        super().train(
            train_dataset, valid_dataset, steps, evaluate_every, save_best_only
        )


class SrganTrainer:
    #
    # TODO: model and optimizer checkpoints
    #
    def __init__(
        self,
        generator,
        discriminator,
        content_loss="VGG54",
        learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5]),
    ):

        if content_loss == "VGG22":
            self.vgg = srgan.vgg_22()
        elif content_loss == "VGG54":
            self.vgg = srgan.vgg_54()
        else:
            raise ValueError("content_loss must be either 'VGG22' or 'VGG54'")

        self.content_loss = content_loss
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

    def train(self, train_dataset, steps=200000):
        pls_metric = Mean()
        dls_metric = Mean()
        step = 0

        for lr, hr in train_dataset.take(steps):
            step += 1

            pl, dl = self.train_step(lr, hr)
            pls_metric(pl)
            dls_metric(dl)

            if step % 50 == 0:
                print(
                    f"{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}"
                )
                pls_metric.reset_states()
                dls_metric.reset_states()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.generator(lr, training=True)

            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss
            disc_loss = self._discriminator_loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(
            perc_loss, self.generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

        return perc_loss, disc_loss

    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss
