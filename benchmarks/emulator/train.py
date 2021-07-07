"""Train emulator."""
import global_config
import emulator.data
import emulator.model
import emulator.training
import emulator.evaluation
import emulator.utils

if __name__ == '__main__':
    """Train and evaluate."""

    conf = emulator.utils.Config(model_name="demo-gru")

    dataset = emulator.data.get_train_data(conf)

    # training_data, training_labels = emulator.data.get_train_data(
    #     data_path=global_config.train_data_path,
    #     labels_path=global_config.train_labels_path
    # )

    model = emulator.model.RNNModel(conf) #Creating Model

    # # Build and train the model
    # model = emulator.model.build(
    #     hidden=128,
    #     dropout=0.2,
    #     lr=0.01,
    #     momentum=0.9
    # )

    emulator.training.train(
        conf,
        model,
        dataset
    )

    # Evaluate the model
    # valid_data, valid_labels = emulator.data.get_train_data(
    #     data_path=global_config.valid_data_path,
    #     labels_path=global_config.valid_labels_path
    # )

    test_data = emulator.data.get_test_data(conf)
    score = emulator.evaluation.evaluate(model, conf, test_data)

    # TODO: Save the final model

    # Print the score
    print('The avg correlation for all test files is: {}'.format(score))
