"""Train emulator."""
import global_config
import emulator.data
import emulator.model
import emulator.training
import emulator.evaluation


if __name__ == '__main__':
    """Train and evaluate."""

    training_data, training_labels = emulator.data.get_train_data(
        data_path=global_config.training_data_path)

    # Build and train the model
    model = emulator.model.build()
    emulator.training.train(model, x=training_data, y=training_labels)

    # Evaluate the model
    valid_data, valid_labels = emulator.data.get_train_data(
        data_path=global_config.validation_data_path)
    score = emulator.evaluation.evaluate(model, x=valid_data, y=valid_labels)

    # TODO: Save the final model

    # Print the score
    print('The model score: {}'.format(score))
