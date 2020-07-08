# Neural-Style-Transfer
Apply Gram matrix and pretrained VGG19 Convolutional Neural Network to analyze and draw a picture in given artistic styles.

# Description:
* Apply Gram matrix and pretrained kernels from VGG19 ConvNet to analyze the Style of the style image using Perceptual Loss and analyze the content in the main image using Texture Loss, and then combine those Losses to define the target of the optimization. Adaptive Momentum optimizer was used, as always, to optimize the image's pixels so that it minimizes the Loss function.
* For deeper details of the algorithm as well as the background math, please take a look at "Algorithm" section and refer to the research paper in the reference.
* If you want to try this algorithm yourself, please go to the link to my Google Colab, which is much more well-documented. Some basic knowledge in Python and Colab/Jupyter Notebook is required to run the algorithm effectively.

# Result:
* Here are some results from this algorithm:
