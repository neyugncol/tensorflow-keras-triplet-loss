from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Input, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Lambda, Add, Subtract, Multiply, Concatenate, BatchNormalization, Dropout
from tensorflow.python.keras.applications import NASNetLarge, InceptionV3, ResNet50, VGG16
from tensorflow.python.keras.optimizers import Adam
import efficientnet.tfkeras as efn


class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1].value, self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs, **kwargs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return None, self.n_classes


class ArcFaceModel(BaseModel):
    def __init__(self, config):
        super(ArcFaceModel, self).__init__(config)
        self.config = config

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
        self.labels = Input(shape=(self.config.num_classes,))

        # backbone = self.supported_backbones[self.config.backbone]
        # self.backbone = backbone(weights=self.config.backbone_weights,
        #                          include_top=False,
        #                          input_tensor=self.inputs)

        # features = [self.backbone.get_layer(name).output for name in self.config.feature_layers]
        # if not features:
        #     features = [self.backbone.output]
        #
        # features = [BatchNormalization()(feature) for feature in features]
        #
        # pooling = self.supported_poolings[self.config.pooling]
        # features = [pooling()(feature) for feature in features]
        #
        # if len(features) > 1:
        #     joint_op = self.supported_joint_ops[self.config.features_joint_op]
        #     features = joint_op()(features)
        # else:
        #     features = features[0]
        #
        # features = Dropout(0.5)(features)
        #
        # features = Dense(units=self.config.embedding_size, kernel_initializer='he_normal',
        #                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(features)
        #
        # features = BatchNormalization()(features)

        # features = self.backbone.output
        #
        # features = BatchNormalization()(features)
        #
        # features = Dropout(0.5)(features)
        #
        # features = Flatten()(features)

        backbone = efn.EfficientNetB2(input_tensor=self.inputs, include_top=False)

        features = backbone.output

        features1 = GlobalAveragePooling2D()(features)
        features2 = GlobalMaxPooling2D()(features)
        features = Concatenate()([features1, features2])

        features = Dense(units=self.config.embedding_size, kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(1e-4))(features)

        features = BatchNormalization()(features)

        outputs = ArcFace(self.config.num_classes, s=15., m=0.35, regularizer=tf.keras.regularizers.l2(1e-4))([features, self.labels])

        self.model = Model(inputs=[self.inputs, self.labels], outputs=outputs, name='arcface_model')

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=self.config.lr),
            metrics=['accuracy']
        )

        features = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(features)

        self.predict_model = Model(inputs=self.inputs, outputs=features, name='arcface_embedded_model')

