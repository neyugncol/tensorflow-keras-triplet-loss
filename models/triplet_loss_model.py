from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Lambda, Add, Subtract, Multiply, Concatenate
from tensorflow.python.keras.applications import NASNetLarge, InceptionV3, ResNet50, VGG16
from tensorflow.python.keras.optimizers import Adam


def _euclidean_pairwise_distance(feature, squared=True):
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
            tf.math.square(tf.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * tf.matmul(feature, tf.transpose(feature))
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

    num_data = tf.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
        tf.ones([num_data]))
    pairwise_distances = tf.math.multiply(pairwise_distances,
                                          mask_offdiagonals)
    return pairwise_distances


def _cosine_pairwise_distance(feature):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    # yapf: disable
    normalized_feature = tf.math.l2_normalize(feature, axis=1)
    pairwise_distances = tf.math.subtract(1.0, tf.linalg.matmul(normalized_feature, normalized_feature, transpose_b=True))
    # yapf: enable

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances = tf.math.maximum(pairwise_distances, 0.0)

    num_data = tf.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
        tf.ones([num_data]))
    pairwise_distances = tf.math.multiply(pairwise_distances,
                                          mask_offdiagonals)
    return pairwise_distances


def _masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.
    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = tf.math.reduce_max(
        tf.math.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums


def _masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.
    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = tf.math.reduce_min(
        tf.math.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums


class TripletLossModel(BaseModel):
    def __init__(self, config):
        super(TripletLossModel, self).__init__(config)
        self.config = config

        if self.config.distance_metric == 'euclidean':
            self.pairwise_distance = _euclidean_pairwise_distance
        elif self.config.distance_metric == 'cosine':
            self.pairwise_distance = _cosine_pairwise_distance

        self.supported_backbones = {
            'nasnet': NASNetLarge,
            'inceptionv3': InceptionV3,
            'resnet50': ResNet50,
            'vgg16': VGG16
        }

        self.supported_poolings = {
            'max': GlobalMaxPooling2D,
            'avg': GlobalAveragePooling2D,
            'flatten': Flatten
        }

        self.supported_joint_ops = {
            'add': Add,
            'subtract': Subtract,
            'multiply': Multiply,
            'concat': Concatenate
        }

        self.build_model()

    def build_model(self):
        self.inputs = Input(shape=(self.config.image_size, self.config.image_size, 3), name='input')

        backbone = self.supported_backbones[self.config.backbone]
        self.backbone = backbone(weights=self.config.backbone_weights,
                                 include_top=False,
                                 input_tensor=self.inputs)

        features = [self.backbone.get_layer(name) for name in self.config.feature_layers]
        if not features:
            features = [self.backbone.layers[-1]]

        pooling = self.supported_poolings[self.config.pooling]
        features = [pooling()(feature) for feature in features]

        if self.config.l2_normalize:
            features = [Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_normalize')(feature) for feature in features]

        if len(features) > 1:
            joint_op = self.supported_joint_ops[self.config.features_joint_op]
            features = joint_op()(features)
        else:
            features = features[0]

        if self.config.fc_layer:
            features = Dense(units=self.config.embedding_size)(features)

        self.model = Model(inputs=self.inputs, outputs=features, name='triplet_loss_model')

        self.model.compile(
            loss=self.get_triplet_loss(),
            optimizer=Adam(lr=self.config.lr),
            metrics=[self.get_pairwise_accuracy()]
        )

    def get_triplet_loss(self):
        margin = tf.constant(self.config.triplet_loss_margin, dtype=tf.float32)
        pairwise_distance = self.pairwise_distance

        def triplet_loss(y_true, y_pred):
            """Computes the triplet loss with semi-hard negative mining.
                Args:
                  y_true: 1-D integer `Tensor` with shape [batch_size] of
                    multiclass integer labels.
                  y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
                    be l2 normalized.
            """
            labels, embeddings = y_true, y_pred
            # Reshape label tensor to [batch_size, 1].
            lshape = tf.shape(labels)
            labels = tf.reshape(labels, [lshape[0], 1])

            # Build pairwise squared distance matrix.
            pdist_matrix = pairwise_distance(embeddings)
            # Build pairwise binary adjacency matrix.
            adjacency = tf.math.equal(labels, tf.transpose(labels))
            # Invert so we can select negatives only.
            adjacency_not = tf.math.logical_not(adjacency)

            batch_size = tf.size(labels)

            # Compute the mask.
            pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
            mask = tf.math.logical_and(
                tf.tile(adjacency_not, [batch_size, 1]),
                tf.math.greater(pdist_matrix_tile,
                                tf.reshape(tf.transpose(pdist_matrix), [-1, 1])))
            mask_final = tf.reshape(
                tf.math.greater(
                    tf.math.reduce_sum(
                        tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True),
                    0.0), [batch_size, batch_size])
            mask_final = tf.transpose(mask_final)

            adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
            mask = tf.cast(mask, dtype=tf.dtypes.float32)

            # negatives_outside: smallest D_an where D_an > D_ap.
            negatives_outside = tf.reshape(
                _masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
            negatives_outside = tf.transpose(negatives_outside)

            # negatives_inside: largest D_an.
            negatives_inside = tf.tile(
                _masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
            semi_hard_negatives = tf.where(mask_final, negatives_outside,
                                           negatives_inside)

            loss_mat = tf.math.add(margin, pdist_matrix - semi_hard_negatives)

            mask_positives = tf.cast(
                adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
                tf.ones([batch_size]))

            # In lifted-struct, the authors multiply 0.5 for upper triangular
            #   in semihard, they take all positive pairs except the diagonal.
            num_positives = tf.math.reduce_sum(mask_positives)

            triplet_loss = tf.math.truediv(
                tf.math.reduce_sum(
                    tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)),
                num_positives)

            return triplet_loss

        return triplet_loss

    def get_pairwise_accuracy(self):
        margin = tf.constant(self.config.triplet_loss_margin, dtype=tf.float32)
        pairwise_distance = self.pairwise_distance

        def pairwise_accuracy(y_true, y_pred):
            """Computes the triplet accuracy.
                Args:
                  y_true: 1-D integer `Tensor` with shape [batch_size] of
                    multiclass integer labels.
                  y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
                    be l2 normalized.
             """
            labels, embeddings = y_true, y_pred
            # Reshape label tensor to [batch_size, 1].
            lshape = tf.shape(labels)
            labels = tf.reshape(labels, [lshape[0], 1])

            batch_size = tf.size(labels)

            # Build pairwise squared distance matrix.
            pdist_matrix = pairwise_distance(embeddings)
            # Build pairwise binary adjacency matrix.
            labels_adjacency = tf.math.equal(labels, tf.transpose(labels))

            embeddings_adjacency = tf.math.less_equal(pdist_matrix, margin)

            pairwise_accuracy = (tf.math.count_nonzero(tf.math.equal(labels_adjacency, embeddings_adjacency), dtype=tf.int32) - batch_size) / (tf.size(labels_adjacency) - batch_size)

            return pairwise_accuracy

        return pairwise_accuracy




