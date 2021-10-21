"""Train emulator."""
import argparse
import os
import logging

import emulator.data
import emulator.model
import emulator.training
import emulator.evaluation
import global_config
import tensorflow as tf


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

    # conf_gru = global_config.Config(
    #     datasets_path=os.path.realpath(
    #             os.path.expanduser(args.datasets_path)),
    #     model_name="demo-gru",
    # )

    conf_reg_l1 = global_config.Config(
        datasets_path=os.path.realpath(
                os.path.expanduser(args.datasets_path)),
        model_name="demo-gru-reg-l1",
    )

    conf_reg_l2 = global_config.Config(
        datasets_path=os.path.realpath(
                os.path.expanduser(args.datasets_path)),
        model_name="demo-gru-reg-l2",
    )


    conf_reg_l2_high = global_config.Config(
        datasets_path=os.path.realpath(
                os.path.expanduser(args.datasets_path)),
        model_name="demo-gru-reg-l2-high",
    )


    # Training
    training_dataset = emulator.data.get_training_dataset(conf_reg_l1)
    
    model1 = emulator.model.RNNModel(conf_reg_l1, reg=True, type=tf.keras.regularizers.L1(0.01))

    for step, loss in emulator.training.train(model1, conf_reg_l1, training_dataset):
        logger.debug('*** {} {}'.format(step, loss))
    
    model1.save_weights(f"../models/{conf_reg_l1.model_name}/ckpts/final_step")
    print(f"Model saved at: ../models/{conf_reg_l1.model_name}/ckpts/final_step")

    model2 = emulator.model.RNNModel(conf_reg_l2, reg=True, type=tf.keras.regularizers.L2(0.01))

    for step, loss in emulator.training.train(model2, conf_reg_l2, training_dataset):
        logger.debug('*** {} {}'.format(step, loss))

    model2.save_weights(f"../models/{conf_reg_l2.model_name}/ckpts/final_step")
    print(f"Model saved at: ../models/{conf_reg_l2.model_name}/ckpts/final_step")

    model3 = emulator.model.RNNModel(conf_reg_l2_high, reg=True, type=tf.keras.regularizers.L2(0.1))

    for step, loss in emulator.training.train(model3, conf_reg_l2_high, training_dataset):
        logger.debug('*** {} {}'.format(step, loss))

    model3.save_weights(f"../models/{conf_reg_l2_high.model_name}/ckpts/final_step")
    print(f"Model saved at: ../models/{conf_reg_l2_high.model_name}/ckpts/final_step")




    # Evaluation
    # validation_dataset = emulator.data.get_validation_dataset(conf_reg_l1)
    # score = emulator.evaluation.evaluate(model, conf, validation_dataset)

    # print('The avg correlation for all test files is: {}'.format(score))
