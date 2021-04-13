from models.DeepLab import Deeplabv3
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


def deeplabv3_mobilenetv2(input):
    """[summary]

    Args:
        input ([type]): [description]

    Returns:
        [type]: [description]
    """
    base_model = Deeplabv3(input_shape=input, classes=1, backbone='mobilenetv2')
    
    #for layer in base_model.layers[:-7]:
    #    layer.trainable = False
    
    return base_model


def deeplabv3_xception(input):
    """[summary]

    Args:
        input ([type]): [description]

    Returns:
        [type]: [description]
    """
    base_model = Deeplabv3(input_shape=input, classes=1, backbone='xception')
    
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    return base_model    


def deepsolar(input):
    """[summary]

    Args:
        input ([type]): [description]

    Returns:
        [type]: [description]
    """
    base_model = tf.keras.applications.InceptionV3(input_shape=[192,192,3], include_top=False)
    base_model.trainable = False
    
    add_model = Sequential()
    add_model.add(base_model)
    add_model.add(GlobalAveragePooling2D())
    add_model.add(Dropout(0.5))
    add_model.add(Dense(nclass, 
                    activation='softmax'))
    
    model = add_model
    return down_stack


def unet_model(output_channels, down_stack):
    """[summary]

    Args:
        output_channels ([type]): [description]
        down_stack ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8,
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]
    
    inputs = tf.keras.layers.Input(shape=[192, 192, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

      # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
          output_channels, 3, strides=2, activation='sigmoid',
          padding='same')  #64x64 -> 192x192
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)    


def vgg19(input):
    """[summary]

    Args:
        input ([type]): [description]

    Returns:
        [type]: [description]
    """
    base_model = tf.keras.applications.VGG19(input_shape=input, include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block1_pool',   # 64x64
        'block2_pool',   # 32x32
        'block3_pool',   # 16x16
        'block4_pool',   # 8x8
        'block5_pool',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False
    
    model = unet_model(1, down_stack)
    
    return model


def resnet152v2(input):
    """[summary]

    Args:
        input ([type]): [description]

    Returns:
        [type]: [description]
    """
    base_model = tf.keras.applications.ResNet152V2(input_shape=input, include_top=False)

    # Use the activations of these layers
    layer_names = [
        'conv1_conv',   # 64x64
        'conv2_block3_1_relu',   # 32x32
        'conv3_block8_1_relu',   # 16x16
        'conv4_block36_1_relu',   # 8x8
        'conv5_block3_2_relu',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False
    
    model = unet_model(1, down_stack)
    
    return model


def resnet101v2(input):
    """[summary]

    Args:
        input ([type]): [description]

    Returns:
        [type]: [description]
    """
    base_model = tf.keras.applications.ResNet101V2(input_shape=input, include_top=False)

    # Use the activations of these layers
    layer_names = [
        'conv1_conv',   # 96x96
        'conv2_block3_1_relu',   # 48x48
        'conv3_block4_1_relu',   # 24x24
        'conv4_block23_1_relu',   # 12x12
        'conv5_block3_2_relu',      # 6x6
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False
    
    model = unet_model(1, down_stack)
    
    return model


def resnet152(input):
    """[summary]

    Args:
        input ([type]): [description]

    Returns:
        [type]: [description]
    """
    base_model = tf.keras.applications.ResNet152(input_shape=input, include_top=False)

    # Use the activations of these layers
    layer_names = [
        'conv1_relu',   # 64x64
        'conv2_block3_2_relu',   # 32x32
        'conv3_block8_2_relu',   # 16x16
        'conv4_block36_2_relu',   # 8x8
        'conv5_block3_2_relu',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False
    
    model = unet_model(1, down_stack)
    
    return model


def resnet50v2(input):
    """[summary]

    Args:
        input ([type]): [description]

    Returns:
        [type]: [description]
    """
    base_model = tf.keras.applications.ResNet50V2(input_shape=input, include_top=False)

    # Use the activations of these layers
    layer_names = [
        'conv1_conv',   # 96x96
        'conv2_block3_1_relu',   # 48x48
        'conv3_block4_1_relu',   # 24x24
        'conv4_block6_1_relu',   # 12x12
        'conv5_block3_2_relu',      # 6x6
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False
    
    model = unet_model(1, down_stack)
    
    return model    


def resnet50(input):
    """[summary]

    Args:
        input ([type]): [description]

    Returns:
        [type]: [description]
    """
    base_model = tf.keras.applications.ResNet50(input_shape=input, include_top=False)

    # Use the activations of these layers
    layer_names = [
        'conv1_relu',   # 64x64
        'conv2_block3_2_relu',   # 32x32
        'conv3_block4_2_relu',   # 16x16
        'conv4_block6_2_relu',   # 8x8
        'conv5_block3_2_relu',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False
    
    model = unet_model(1, down_stack)
    
    return model    


def mobile_net(input):
    """[summary]

    Args:
        input ([type]): [description]

    Returns:
        [type]: [description]
    """
    base_model = tf.keras.applications.MobileNet(input_shape=input, alpha = 0.5, include_top=False)

    # Use the activations of these layers
    layer_names = [
        'conv_pw_1_relu',   # 64x64
        'conv_pw_3_relu',   # 32x32
        'conv_pw_5_relu',   # 16x16
        'conv_pw_11_relu',  # 8x8
        'conv_pw_13_relu',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False
    
    model = unet_model(1, down_stack)
    
    return model    


def mobile_net_v2(input):
    """[summary]

    Args:
        input ([type]): [description]

    Returns:
        [type]: [description]
    """
    base_model = tf.keras.applications.MobileNetV2(input_shape=input, include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False
    
    model = unet_model(1, down_stack)
    
    return model