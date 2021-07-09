"""Train emulator."""
import argparse
import os
import logging

import emulator.data
import emulator.model
import emulator.training
import emulator.evaluation
import global_config


if __name__ == '__main__':
    """Train and evaluate."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--datasets_path', type=str,
        help='Path to the datasets directory')
    args = parser.parse_args()

    if not args.datasets_path:
        raise Exception('You must specify datasets path.')

    conf = global_config.Config(
        datasets_path=os.path.realpath(
                os.path.expanduser(args.datasets_path)),
        model_name="demo-gru",
    )

    # Training
    training_dataset = emulator.data.get_training_dataset(conf)
    model = emulator.model.RNNModel(conf)

    for step, loss in emulator.training.train(model, conf, training_dataset):
        logger.debug('*** {} {}'.format(step, loss))

    # Evaluation
    validation_dataset = emulator.data.get_validation_dataset(conf)
    score = emulator.evaluation.evaluate(model, conf, validation_dataset)

    print('The avg correlation for all test files is: {}'.format(score))
