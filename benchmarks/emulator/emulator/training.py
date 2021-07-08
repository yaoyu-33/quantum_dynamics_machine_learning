"""Neural network training."""
import tensorflow as tf
import datetime
import time
import os
from emulator.optimization import create_optimizer
import emulator.utils


# def train(model, x, y, epochs=10, batch_size=32):
#     """Train neural network.

#     Args:
#         model (tensorflow.keras.Model): model to be trained
#         x (numpy.array): training data
#         y (numpy.array): training labels
#         epochs (int): number of epochs
#         batch_size (int): batch size

#     Returns:
#         (tensorflow.keras.Model): trained model
#     """



#     # TODO: Update the training procedure.

#     model.fit(
#         x,
#         y,
#         batch_size=batch_size,
#         epochs=epochs,
#         verbose=0,
#     )

   
def train(config, model, dataset):
    """ Main training loop """
    @tf.function #Uses XLA for faster training
    def train_step(feature, target, model, optimizer):
        mse = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            predict = model(feature, True)
            loss = mse(target, predict)
        gradients = tape.gradient(loss, model.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(g, config.clip_norm) for g in gradients]
        optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
        return loss

    #Seed for replication
    tf.random.set_seed(config.seed)

    #Logging during training
    log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") if config.log_name is None else config.log_name
    train_log_dir = config.log_dir + "/" + log_name
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    f = open(train_log_dir + "/" + "config.txt", "w")
    f.write(str(config.__dict__))
    f.close()

    # #Load dataset from config
    # dataset = iter(get_dataset(config))

    #Optimizer
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

    #Training for every step
    while step < config.num_train_steps:
        feature, target = next(dataset)
        loss = train_step(feature, target, model, optimizer)
        metrics["train_loss"].update_state(loss)
        step += 1
        if step % config.logging_steps == 0:
            print("Step: {:6d}, Loss: {:.8f}, Time per logging: {:.2f}s, Elapsed: {:}, ETA: {:}".format(
                step,
                metrics["train_loss"].result(),
                time.time() - logging_time,
                emulator.utils.get_readable_time(time.time() - start_time),
                emulator.utils.get_readable_time((time.time() - start_time) / step * (config.num_train_steps - step))), flush=True)
            with train_summary_writer.as_default():
                for k in metrics:
                    tf.summary.scalar(k, metrics[k].result(), step=step)
                    metrics[k].reset_states()
            logging_time = time.time()
        if step % config.save_ckpt_steps == 0 and step != config.num_train_steps:
            model.save_weights(os.path.join(config.ckpts, "step_" + str(step)))
            print("Model weights saved at ", os.path.join(config.ckpts, "step_" + str(step)))

    model.save_weights(os.path.join(config.ckpts, "final_step"))
    print("Model Trained!")
    print("Model weights saved at ", os.path.join(config.ckpts, "final_step"))
