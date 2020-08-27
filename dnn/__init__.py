from tensorflow.keras.layers import Conv3D, Conv3DTranspose, BatchNormalization, Dropout
from tensorflow.keras.layers import Input, concatenate, MaxPooling3D, Activation, Reshape
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


def conv_block_3d(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv3D(filters=nfilters, kernel_size=(size, size, size), padding=padding)(tensor)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv3D(filters=nfilters, kernel_size=(size, size, size), padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    return x


def deconv_block_3d(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2, 2)):
    y = Conv3DTranspose(nfilters, kernel_size=(size, size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=4)
    y = conv_block_3d(y, nfilters)
    return y


def u_net_3d(img_height: int, img_width: int, bands: int,
             time_steps: int, nclasses: int, filters=16) -> Model:
    input_layer = Input(shape=(time_steps, img_height, img_width, bands), name='image_input')
    # First Conv Block
    conv1 = conv_block_3d(input_layer, nfilters=filters)
    conv1 = BatchNormalization()(conv1)
    conv2 = conv_block_3d(conv1, nfilters=filters * 2)
    conv2 = BatchNormalization()(conv2)
    conv2_out = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
    # Second Conv Block
    conv3 = conv_block_3d(conv2_out, nfilters=filters * 4)
    conv3 = BatchNormalization()(conv3)
    conv4 = conv_block_3d(conv3, nfilters=filters * 8)
    conv4 = BatchNormalization()(conv4)
    conv4_out = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv4)
    # Third Conv Block
    conv4_out = Dropout(0.5)(conv4_out)
    conv4_out = BatchNormalization()(conv4_out)
    conv5 = conv_block_3d(conv4_out, nfilters=filters * 16)
    conv5 = Dropout(0.5)(conv5)
    # Deconv Block
    deconv6 = deconv_block_3d(conv5, residual=conv4, nfilters=filters * 8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block_3d(deconv6, residual=conv2, nfilters=filters * 4)
    deconv7 = Dropout(0.5)(deconv7)
    # Post Conv/Deconv Processing Block
    conv6 = Conv3D(filters=nclasses, kernel_size=(time_steps, 1, 1), padding='valid')(deconv7)
    conv6 = BatchNormalization()(conv6)
    shapes = conv6.get_shape()
    conv6 = Reshape((shapes[2], shapes[3], shapes[4]))(conv6)
    output = Activation("elu")(conv6)
    model = Model(inputs=input_layer, outputs=output, name='Unet')
    model.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
