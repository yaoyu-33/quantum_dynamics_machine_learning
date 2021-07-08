"""Neural network training."""
import tensorflow as tf
import logging
import time
from emulator.optimization import create_optimizer
import emulator.utils


def train(model, config, dataset):
    """Training loop.

    Args:
        model (tensorflow.keras.Model): neural network
        config (global_config.Config): config object with training parameters
        dataset (tensorflow.data.Dataset): Training dataset
    Returns:
        None
    """
    @tf.function  # Uses XLA for faster training
    def train_step(feature, target, model, optimizer):
        mse = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            predict = model(feature, True)
            loss = mse(target, predict)
        gradients = tape.gradient(loss, model.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(g, config.clip_norm)
                             for g in gradients]
        optimizer.apply_gradients(zip(clipped_gradients,
                                      model.trainable_variables))
        return loss

    # Optimizer
    optimizer, lr_schedule = create_optimizer(
        init_lr=config.learning_rate,
        num_train_steps=config.num_train_steps,
        num_warmup_steps=config.num_warmup_steps,
        weight_decay_rate=config.weight_decay_rate,
        power=config.lr_decay_power,
        adam_beta1=config.opt_beta_1,
        adam_beta2=config.opt_beta_2,
        min_lr_ratio=config.end_lr / config.learning_rate)
    tf.config.optimizer.set_jit(True)

    metrics = {"train_loss": tf.keras.metrics.Mean(name="train_loss")}
    step = 0
    start_time = logging_time = time.time()

    # Training for every step
    while step < config.num_train_steps:
        feature, target = next(dataset)
        loss = train_step(feature, target, model, optimizer)
        metrics["train_loss"].update_state(loss)
        step += 1
        if step % config.hyper_step_size == 0:
            logging.info(
                """
                    Step: {:6d},
                    Loss: {:.8f},
                    Time per log: {:.2f}s,
                    Elapsed: {:},
                    ETA: {:}
                """
                .format(
                    step,
                    metrics["train_loss"].result(),
                    time.time() - logging_time,
                    emulator.utils.get_readable_time(time.time() - start_time),
                    emulator.utils.get_readable_time(
                        (time.time() - start_time) /
                        step * (config.num_train_steps - step))), flush=True)
            yield step, float(metrics["train_loss"].result())

            for k in metrics:
                metrics[k].reset_states()
            logging_time = time.time()

    logging.info("Model Fully Trained!")
    yield step, float(metrics["train_loss"].result())
