"""Global task config."""
import os


class Config:
    """Task config."""

    def __init__(self, datasets_path, **kwargs):
        """ Configuration Settings for Training and Models"""

        # Read/write Settings
        self.training_files = os.path.join(datasets_path, 'demo/tfrecords/*')
        self.model_name = "demo"
        self.models_dir = "models/"
        self.validation_files = os.path.join(
            datasets_path, 'test/full_test_E0_0.0to9.0_BH_0.0to14.0/*.txt')
        self.log_name = None

        # Training Settings
        self.train_batch = 128
        self.input_frames = 4
        self.input_channels = 3
        self.output_channels = 2
        self.window_size = 23
        self.dropout_rate = 0.
        self.hidden_size = 690
        self.num_train_steps = 90000  # up to 900_000

        # Optimizer Settings
        self.learning_rate = 1e-3
        self.lr_decay_power = 1.0
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 9000
        self.opt_beta_1 = 0.9
        self.opt_beta_2 = 0.999
        self.end_lr = 1e-6
        self.clip_norm = 1.0

        # Misc
        self.seed = 711
        self.hyper_step_size = 10000  # at least 10_000
        self.save_ckpt_steps = 1000000

        self.update(kwargs)
        self.model_dir = os.path.join(self.models_dir, self.model_name)
        self.log_dir = os.path.join(self.model_dir, "logs")
        self.ckpts = os.path.join(self.model_dir, "ckpts")

    def update(self, kwargs):
        # TODO: Update Docstrings
        for k, v in kwargs.items():
            if v is not None:
                self.__dict__[k] = v
