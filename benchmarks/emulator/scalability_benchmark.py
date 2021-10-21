"""Scalability Benchmark."""
import argparse
import os.path

import tensorflow

import ray.tune
import ray.tune.schedulers

import global_config
import emulator.data
import emulator.model
import emulator.training
import emulator.evaluation


class Emulator(ray.tune.Trainable):
    """Emulator."""

    def setup(self, config):
        """Setup."""
        conf = global_config.Config(
            datasets_path=config['datasets_path'],
            model_name="demo-gru",
            hidden_size=config['hidden'],
            # dropout_rate=config['dropout'],
            # learning_rate=config['lr'],
            # opt_beta_1=config['momentum'],
            # weight_decay_rate=config['weight_decay_rate']
        )

        self.train_dataset = emulator.data.get_training_dataset(conf)
        self.valid_dataset = emulator.data.get_validation_dataset(conf)
        self.model = emulator.model.RNNModel(conf)
        self.conf = conf

    def step(self):
        """One step of training."""
        next(emulator.training.train(self.model,
                                     self.conf,
                                     self.train_dataset))
        score = emulator.evaluation.evaluate(
            self.model, self.conf, self.valid_dataset)

        return {"mean_correlation": score}

    def save_checkpoint(self, checkpoint_dir):
        """Save the model's checkpoint."""
        file_path = checkpoint_dir + "/model"
        print('***>>>', checkpoint_dir, file_path)
        self.model.save(file_path)
        return file_path

    def load_checkpoint(self, path):
        """Load the checkpoint."""
        del self.model
        print('***<<<', path)
        self.model = tensorflow.keras.models.load_model(path)


if __name__ == "__main__":
    """Run."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--cpu_number', type=int,
        help='Number of CPUs.')
    parser.add_argument(
        '-d', '--datasets_path', type=str,
        help='Path to the datasets directory')
    parser.add_argument(
        '-g', '--gpu_number', type=int, default=0,
        help='Number of GPUs.')
    args = parser.parse_args()

    if not args.datasets_path:
        raise Exception('You must specify datasets path.')

    # CAVEAT: If you want to use scalability_benchmark_multi_nodes.job
    #         You must use `ray.init('auto')` instead of
    #         ray.init(num_cpus=..., num_gpus=...)
    ray.init(
        num_cpus=args.cpu_number,
        num_gpus=args.gpu_number,
        log_to_driver=True  # Change to False to hide clients' logs
    )

    scheduler = ray.tune.schedulers.AsyncHyperBandScheduler(
        time_attr="training_iteration",
        max_t=100,
        grace_period=5
    )

    results = ray.tune.run(
        Emulator,
        name="pbt_emulator_tuning",
        scheduler=scheduler,
        metric="mean_correlation",
        mode="max",
        stop={
            "training_iteration": 10  # TODO: In the final run increase this
        },
        num_samples=4,  # TODO: In the final run you might increase it
        config={
            'datasets_path': os.path.realpath(
                os.path.expanduser(args.datasets_path)),
            'hidden': ray.tune.randint(60, 70),
            # 'dropout': ray.tune.uniform(0, 0.5),
            # 'lr': ray.tune.loguniform(1e-5, 1e-1), 
            # 'momentum': ray.tune.uniform(0.6, 1),
            # 'weight_decay_rate': ray.tune.uniform(0.01,0.05)
        },
        resources_per_trial={'cpu': 1, 'gpu': 1 if args.gpu_number else 0},
    )

    print("Best hyperparameters found were: ", results.best_config)
