"""Scalability Benchmark."""
import argparse
import numpy
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
            data_path=global_config.training_data_path)
        self.valid_x, self.valid_y = emulator.data.get_validation_data(
            data_path=global_config.validation_data_path)
        self.model = emulator.model.build()

    def step(self):
        """One step of training."""
        emulator.training.train(
            self.model, self.train_x, self.train_y, epochs=1)

        score = emulator.evaluation.evaluate(
            self.model, self.valid_x, self.valid_y)

        return {"mean_accuracy": score}

    def save_checkpoint(self, checkpoint_dir):
        file_path = checkpoint_dir + "/model"
        self.model.save(file_path)
        return file_path

    def load_checkpoint(self, path):
        del self.model
        self.model = tensorflow.keras.models.load_model(path)


if __name__ == "__main__":
    """Run."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--cpu_number', help='Number of CPUs.', type=int)
    args = parser.parse_args()

    ray.init(num_cpus=args.cpu_number, log_to_driver=False)

    pbt = ray.tune.schedulers.PopulationBasedTraining(
        perturbation_interval=2,
        hyperparam_mutations={
            "dropout": lambda: numpy.random.uniform(0, 1),
            "lr": lambda: numpy.random.uniform(0.001, 0.1),
            "hidden": lambda: numpy.random.randint(32, 256)
        })

    results = ray.tune.run(
        Emulator,
        name="pbt_emulator_tuning",
        scheduler=pbt,
        metric="mean_accuracy",
        mode="max",
        stop={
            "training_iteration": 3  # TODO: In the final run increase this
        },
        num_samples=12,  # TODO: In the final run increase to 16 or 32
        config={
            "sth": 0.9
        })

    print("Best hyperparameters found were: ", results.best_config)
