# @author Anh Nhu - 2020

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio
import cv2
# import png

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



"""
Return the activations in different convolutional layers in VGG19 of img
"""
def model_activations(img, input):

    model = tf.keras.applications.VGG19(include_top = False, weights = "imagenet")
    model.trainable = False # we will not change the paramenters of the VGG19
    # model.summary()

    res = {}

    """
    Define layers of VGG19 from Tensorflow
    """
    input = model.get_layer(name = input)(img)

    block1_conv1_features = model.get_layer(name = "block1_conv1")(input)
    block1_conv2_features = model.get_layer(name = "block1_conv2")(block1_conv1_features)
    block1_pool_features  = model.get_layer(name = "block1_pool")(block1_conv2_features)

    block2_conv1_features = model.get_layer(name = "block2_conv1")(block1_pool_features)
    block2_conv2_features = model.get_layer(name = "block2_conv2")(block2_conv1_features)
    block2_pool_features  = model.get_layer(name = "block2_pool")(block2_conv2_features)

    block3_conv1_features = model.get_layer(name = "block3_conv1")(block2_pool_features)
    block3_conv2_features = model.get_layer(name = "block3_conv2")(block3_conv1_features)
    block3_conv3_features = model.get_layer(name = "block3_conv3")(block3_conv2_features)
    block3_conv4_features = model.get_layer(name = "block3_conv4")(block3_conv3_features)
    block3_pool_features  = model.get_layer(name = "block3_pool")(block3_conv4_features)

    block4_conv1_features = model.get_layer(name = "block4_conv1")(block3_pool_features)
    block4_conv2_features = model.get_layer(name = "block4_conv2")(block4_conv1_features)
    block4_conv3_features = model.get_layer(name = "block4_conv3")(block4_conv2_features)
    block4_conv4_features = model.get_layer(name = "block4_conv4")(block4_conv3_features)
    block4_pool_features = model.get_layer(name  = "block4_pool")(block4_conv4_features)

    block5_conv1_features = model.get_layer(name = "block5_conv1")(block4_conv4_features)
    block5_conv2_features = model.get_layer(name = "block5_conv2")(block5_conv1_features)
    block5_conv3_features = model.get_layer(name = "block5_conv3")(block5_conv2_features)
    block5_conv4_features = model.get_layer(name = "block5_conv4")(block5_conv3_features)
    block5_pool_features  = model.get_layer(name = "block5_pool")(block5_conv4_features)

    res["b1_conv1_activation"] = block1_conv1_features
    res["b1_conv2_activation"] = block1_conv2_features
    res["b1_pool_activation"]  = block1_pool_features

    res["b2_conv1_activation"] = block2_conv1_features
    res["b2_conv2_activation"] = block2_conv2_features
    res["b2_pool_activation"]  = block2_pool_features

    res["b3_conv1_activation"] = block3_conv1_features
    res["b3_conv2_activation"] = block3_conv2_features
    res["b3_conv3_activation"] = block3_conv3_features
    res["b3_conv4_activation"] = block3_conv4_features
    res["b3_pool_activation"]  = block3_pool_features

    res["b4_conv1_activation"] = block4_conv1_features
    res["b4_conv2_activation"] = block4_conv2_features
    res["b4_conv3_activation"] = block4_conv3_features
    res["b4_conv4_activation"] = block4_conv4_features
    res["b4_pool_activation"]  = block4_pool_features

    res["b5_conv1_activation"] = block5_conv1_features
    res["b5_conv2_activation"] = block5_conv2_features
    res["b5_conv3_activation"] = block5_conv3_features
    res["b5_conv4_activation"] = block5_conv4_features
    res["b5_pool_activation"]  = block5_pool_features

    return res

"""
Define the Content Loss / Texture Loss
@param activation_product: activations of the Generated Image at a given convolutional layer
@param activation_content: activations of the Content Image at the corresponding convolutional layer 
"""
def Loss_C(activation_product, activation_content):
    # index 0 is the number of images by default
    n_H = activation_product.shape[1] # vertical dimension of a channel in current activation layer
    n_W = activation_product.shape[2] # horizontal dimension of a channel in current activation layer
    n_C = activation_product.shape[3] # number of channels in current activation layer

    loss = tf.reduce_sum(tf.pow(activation_product - activation_content, 2))
    loss = (1 / (4 * n_H * n_W * n_C)) * loss

    return loss

"""
Define the Perceptual Loss / Feature Loss for 1 layer
The core of the algorithm
@param activation_product: activations of the Generated Image at a given convolutional layer
@param activation_perceptual: activations of the Style Image at the corresponding convolutional layer
"""
def Loss_P_layer(activation_product, activation_perceptual):
    # index 0 is the number of images by default
    n_H = activation_product.shape[1] # vertical dimension of a channel in current activation layer
    n_W = activation_product.shape[2] # horizontal dimension of a channel in current activation layer
    n_C = activation_product.shape[3] # number of channels in current activation layer

    # The shape of the activation layer in the CNN
    # Both activation_product and activation_perceptual must have the same dimension (bc of the same corresponding layer)
    layer_shape = activation_product.shape

    # unroll matrix for computational efficiency, I will explain it more clearly in the notebook / github
    # Note: Use tf.reshape() instead of np.reshape()
    unroll_product = tf.reshape(activation_product, (1, layer_shape[1] * layer_shape[2], layer_shape[3]))
    unroll_perceptual = tf.reshape(activation_perceptual, (1, layer_shape[1] * layer_shape[2], layer_shape[3]))

    # Define Gram Matrices
    # Note: Core of the the entire Algorithm
    G_product = tf.matmul(tf.transpose(unroll_product[0]), unroll_product[0])
    G_perceptual = tf.matmul(tf.transpose(unroll_perceptual[0]), unroll_perceptual[0])

    Loss = tf.reduce_sum(tf.pow(G_product - G_perceptual, 2))
    Loss = (1 / (2 * n_H * n_W * n_C)**2) * Loss

    return Loss 




"""
Draw the picture
"""
def Draw(epoch, opt):
  layer_1_C = 'b4_conv2_activation'

  layer_1_P = 'b1_conv1_activation'
  layer_2_P = 'b2_conv1_activation'
  layer_3_P = 'b3_conv1_activation'
  layer_4_P = 'b4_conv1_activation'
  layer_5_P = 'b5_conv1_activation'

  with tf.GradientTape() as tape:
      # @author Anh Nhu
      # Layer activations of the product through VGG19
      product_activation = model_activations(product, input = "input_" + str(epoch+3))

      # Define Loss function / Optimization goal
      loss_C = Loss_C(product_activation[layer_1_C], activation_C[layer_1_C])
      loss_P = 0.05 * Loss_P_layer(product_activation[layer_1_P], activation_P[layer_1_P]) + 0.05 * Loss_P_layer(product_activation[layer_2_P], activation_P[layer_2_P]) + 0.1 * Loss_P_layer(product_activation[layer_3_P], activation_P[layer_3_P]) + 0.3 * Loss_P_layer(product_activation[layer_4_P], activation_P[layer_4_P]) + 0.7 * Loss_P_layer(product_activation[layer_5_P], activation_P[layer_5_P])
      loss = gamma_1 * loss_C + gamma_2 * loss_P
      
      # Compute Gradient / Direction with highest rate of change
      grad = tape.gradient(loss, product)
      # Apply Gradient on the pixels
      opt.apply_gradients([(grad, product)])

      # Clip the pixel values that fall outside the range of [0,1]
      product.assign(tf.clip_by_value(product, clip_value_min=0.0, clip_value_max=1.0))

      
      #show resulting image after each epoch
      plt.imshow(product[0,:,:,:])
      plt.title("Drawing... Epoch " + str(epoch))
      plt.show()
      print(loss)
      

"""
Create an empty template to start drawing
"""
def create_template():
    img_temp = np.ones((image_shape[0], image_shape[1], image_shape[2])) * 255
    blank_paper = Image.fromarray(img_temp.astype('uint8')).convert('RGB')
    blank_paper = tf.keras.preprocessing.image.img_to_array(blank_paper)
    blank_paper = np.expand_dims(blank_paper, axis = 0)
    blank_paper = tf.Variable(blank_paper)

    return blank_paper



# desired image size
image_shape = (512, 512, 3)


# Load and display texture image
image_C = tf.keras.preprocessing.image.load_img("/content/New York.jpg", target_size = image_shape)

"""
Preprocess the input
"""
img_C = tf.keras.preprocessing.image.img_to_array(image_C) # convert image to array to feed into CNN
img_C = np.expand_dims(img_C, axis = 0) # we must have 1 dimension for the number of images

img_C[0,:,:,:] = img_C[0,:,:,:] / 255.


plt.imshow(image_C)
plt.title("Sample Image")
plt.show()

plt.imshow(img_C[0,:,:,:])
plt.title("Content Image")
plt.show()

tf.keras.preprocessing.image.save_img(path = '/content/C.jpg', x = img_C[0,:,:,:])

#########################################################################################################

# Load and display perceptual image
image_P = tf.keras.preprocessing.image.load_img("/content/Starry Night.jpg", target_size = image_shape)

"""
Preprocess the input
"""
img_P = tf.keras.preprocessing.image.img_to_array(image_P) # convert image to array to feed into CNN
img_P = np.expand_dims(img_P, axis = 0) # we must have 1 dimension for the number of images

img_P[0,:,:,:] = img_P[0,:,:,:] / 255.


plt.imshow(image_P)
plt.title("Sample Image")
plt.show()

plt.imshow(img_P[0,:,:,:])
plt.title("Perceptual Image")
plt.show()

#########################################################################################################

# activations of the Content Image in different layers in CNN, these activations are fixed
activation_C = model_activations(img_C, input = "input_1")
activation_P = model_activations(img_P, input = "input_2")

# product initially is a blank paper

# you should change this to True, if you start drawing in the first epoch
# I need several different epochs, each start with the past result to avoid GPU out of memory issue
blank = False 

if blank == True:
  product = create_template()
else:
  product = tf.keras.preprocessing.image.load_img("/content/Image.jpg", target_size = image_shape)
  product = tf.keras.preprocessing.image.img_to_array(product)
  product = product / 255.
  product = np.expand_dims(product, axis = 0)
  product = tf.Variable(product)



plt.imshow(product[0,:,:,:])
plt.title("Start")
plt.show()

gamma_1 = 1e2 # how much do we care about the texture
gamma_2 = 1e2 # how much do we care about the perceptual


# Draw part
for i in range(0, 160):
    Draw(i, opt = tf.optimizers.Adam(learning_rate = 0.001))



# @author Anh Nhu - 2020
# Display the final product
prod = product.numpy()

# save image for the next training; I have to do this due to limitation of GPU memory, else it will throw OOM error
tf.keras.preprocessing.image.save_img(path = '/content/Image.jpg', x = prod[0,:,:,:])

prod = tf.keras.preprocessing.image.array_to_img(prod[0,:,:,:])

plt.imshow(prod)
plt.title("Final")
plt.show()
