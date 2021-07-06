"""Scalability Benchmark."""
import argparse
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
        self.train_x, self.train_y = emulator.data.get_train_data(
            data_path=global_config.train_data_path,
            labels_path=global_config.train_labels_path
        )
        self.valid_x, self.valid_y = emulator.data.get_validation_data(
            data_path=global_config.valid_data_path,
            labels_path=global_config.valid_labels_path
        )
        self.model = emulator.model.build(
            hidden=config['hidden'],
            dropout=config['dropout'],
            lr=config['lr'],
            momentum=config['momentum'],
        )

    def step(self):
        """One step of training."""
        emulator.training.train(
            self.model, self.train_x, self.train_y, epochs=1)

        score = emulator.evaluation.evaluate(
            self.model, self.valid_x, self.valid_y)

        return {"mean_accuracy": score}

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
        '-c', '--cpu_number', help='Number of CPUs.', type=int)
    parser.add_argument(
        '-g', '--gpu_number', help='Number of GPUs.', type=int, default=0)
    args = parser.parse_args()

    ray.init(
        num_cpus=args.cpu_number,
        num_gpus=args.gpu_number,
        log_to_driver=True
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
        metric="mean_accuracy",
        mode="max",
        stop={
            "training_iteration": 6  # TODO: In the final run increase this
        },
        num_samples=4,  # TODO: In the final run you might increase it
        config={
            'hidden': ray.tune.randint(32, 512),
            'dropout': ray.tune.uniform(0, 1),
            'lr': ray.tune.loguniform(1e-5, 1e-1),
            'momentum': ray.tune.uniform(0.5, 1),
        },
        checkpoint_freq=2,
        checkpoint_at_end=True,
        resources_per_trial={'cpu': 2, 'gpu': 1 if args.gpu_number else 0}
    )

    print("Best hyperparameters found were: ", results.best_config)
