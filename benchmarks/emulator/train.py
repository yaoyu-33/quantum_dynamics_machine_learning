"""Train emulator."""
import argparse
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
        datasets_path=args.datasets_path,
        model_name="demo-gru",
    )

    # Training
    training_dataset = emulator.data.get_training_dataset(conf)
    model = emulator.model.RNNModel(conf)
    emulator.training.train(conf, model, training_dataset)

    # Evaluation
    validation_dataset = emulator.data.get_validation_dataset(conf)
    score = emulator.evaluation.evaluate(model, conf, validation_dataset)

    print('The avg correlation for all test files is: {}'.format(score))
