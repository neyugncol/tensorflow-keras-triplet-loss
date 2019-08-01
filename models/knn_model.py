from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Flatten, Dense, Lambda
from tensorflow.python.keras.applications import NASNetLarge, InceptionV3, ResNet50, VGG16
from tensorflow.python.keras.optimizers import Adam


def _reference_pairwise_distance(feature, reference, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    # yapf: disable
    pairwise_distances_squared = tf.math.add(
        tf.math.reduce_sum(
            tf.math.square(feature),
            axis=[1],
            keepdims=True),
        tf.math.reduce_sum(
            tf.math.square(tf.transpose(reference)),
            axis=[0],
            keepdims=True)) - 2.0 * tf.matmul(feature, tf.transpose(reference))
    # yapf: enable

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared,
                                                 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared +
            tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = tf.math.multiply(
        pairwise_distances,
        tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32))

    return pairwise_distances


class KNNModel(BaseModel):
    def __init__(self, config, embedding_model):
        super(KNNModel, self).__init__(config)
        self.config = config
        self.embedding_model = embedding_model

        self.build_model()

    def build_model(self):
        self.inputs = self.embedding_model.input
        self.embedding_references = Input(shape=(self.config.embedding_size), name='embedding_references')

        self.embeddings = self.embedding_model.output

        pairwise_distance = _reference_pairwise_distance(self.embeddings, self.embedding_references)

        prediction = tf.math.argmin(pairwise_distance, axis=1)

        self.model = Model(inputs=[self.inputs, self.embedding_references], outputs=prediction, name='knn_model')
