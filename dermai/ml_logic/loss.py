import tensorflow as tf
from tensorflow.keras.backend import epsilon
from tensorflow.keras.utils import register_keras_serializable

DANGEROUS_CLASSES = [1, 4]
DANGEROUS_WEIGHT = 3.0
NORMAL_WEIGHT = 1.0

@register_keras_serializable(package="CustomLosses")
# @keras.utils.register_keras_serializable(package="CustomLosses")
def weighted_categorical_crossentropy(y_true, y_pred):
    """
    Creates a weighted categorical crossentropy loss that penalizes
    misclassification of dangerous classes more heavily to maximize recall.

    Args:
        dangerous_classes: List of class indices to weight more heavily (e.g., [1, 4] for bcc and mel)
        dangerous_weight: Weight multiplier for dangerous classes (higher = prioritize recall)
        normal_weight: Weight for non-dangerous classes

    Returns:
        Loss function compatible with model.compile()

    Usage:
        model.compile(
            optimizer=Adam(learning_rate=INITIAL_LR),
            loss=weighted_categorical_crossentropy(dangerous_classes=[1, 4], dangerous_weight=3.0),
            metrics=[...]
        )
    """

    # Clip predictions to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, epsilon(), 1 - epsilon())

    # Calculate standard categorical crossentropy
    cce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

    # Get the true class index for each sample
    true_class_idx = tf.argmax(y_true, axis=-1)

    # Create weight tensor: start with normal_weight for all samples
    weights = tf.ones_like(true_class_idx, dtype=tf.float32) * NORMAL_WEIGHT

    # Apply dangerous_weight to samples from dangerous classes
    for dangerous_class in DANGEROUS_CLASSES:
        weights = tf.where(
            tf.equal(true_class_idx, dangerous_class),
            DANGEROUS_WEIGHT,
            weights
        )

    # Apply weights to loss
    weighted_loss = cce * weights

    return weighted_loss
