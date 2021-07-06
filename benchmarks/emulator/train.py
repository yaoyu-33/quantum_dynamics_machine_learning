"""Train emulator."""
import global_config
import emulator.data
import emulator.model
import emulator.training
import emulator.evaluation


if __name__ == '__main__':
    """Train and evaluate."""

    training_data, training_labels = emulator.data.get_train_data(
        data_path=global_config.train_data_path,
        labels_path=global_config.train_labels_path
    )

    # Build and train the model
    model = emulator.model.build(
        hidden=128,
        dropout=0.2,
        lr=0.01,
        momentum=0.9
    )
    emulator.training.train(
        model,
        x=training_data,
        y=training_labels,
        epochs=10,
        batch_size=32
    )

    # Evaluate the model
    valid_data, valid_labels = emulator.data.get_train_data(
        data_path=global_config.valid_data_path,
        labels_path=global_config.valid_labels_path
    )
    score = emulator.evaluation.evaluate(model, x=valid_data, y=valid_labels)

    # TODO: Save the final model

    # Print the score
    print('The model score: {}'.format(score))
