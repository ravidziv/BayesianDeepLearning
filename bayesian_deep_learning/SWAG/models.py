import tensorflow as tf


class SWAG_model(tf.keras.Model):

    def __init__(self, base_model: tf.keras.Model):
        """
        Wrapper for tf.keras model that include the base model and the swag model
        :param base_model: model to calculate SWAG in its weights
        """
        super(SWAG_model, self).__init__()
        self.base_model = base_model
        self.swag_model = tf.keras.models.clone_model(base_model, input_tensors=None, clone_function=None)
        # todo - make it configurable
        self.swag_loss = tf.keras.metrics.CategoricalCrossentropy(from_logits=True, name='swag_loss')
        self.swag_accuracy = tf.keras.metrics.CategoricalAccuracy(name="swag_accuracy")

    def call(self, inputs, training=None, mask=None):
        output = self.base_model(inputs, training=training)
        return output

    def update_swag_metrics(self, data):
        x, y = data
        y_pred = self.swag_model(x)
        for m in self.metrics:
            if 'swag' in m.name:
                m.update_state(y, y_pred)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update the weights of the base model
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.update_swag_metrics(data)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        return {'base_model': self.base_model}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Update the parameters of SWAG
        self.update_swag_metrics(data)
        return {m.name: m.result() for m in self.metrics}
