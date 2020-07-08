# Neural-Style-Transfer
Apply Gram matrix and pretrained VGG19 Convolutional Neural Network to analyze and draw a picture in given artistic styles.

# Description:
* Apply Gram matrix and pretrained kernels from VGG19 ConvNet to analyze the Style of the style image using Perceptual Loss and analyze the content in the main image using Texture Loss, and then combine those Losses to define the target of the optimization. Adaptive Momentum optimizer was used, as always, to optimize the image's pixels so that it minimizes the Loss function.
* For deeper details of the algorithm as well as the background math, please take a look at "Algorithm" section and refer to the research paper in the reference.
* If you want to try this algorithm yourself, please go to the link to my Google Colab, which is much more well-documented. Some basic knowledge in Python and Colab/Jupyter Notebook is required to run the algorithm effectively.

# Result:
Here are some results from this algorithm:<br><br>
**1) Scene (Paris)**

**Original Image:**<br><br>
<img src = "Results/0. Paris.jpg"><br><br><br>
**Art 1: Oil**<br>
<img src = "Results/1. Oil.jpg"><br><br><br>
**Result 1: Oil picture of Paris**<br><br>
<img src = "Results/1. Paris_Oil.jpg"><br><br><br>
**Art 2: Van Gogh**<br>
<img src = "Results/2. Starry Night.jpg"><br><br><br>
**Result 1: Paris by Van Gogh**<br><br>
<img src = "Results/2.1 Paris_Starry.jpg"><br><br><br>
**Art 3: Futurism**<br><br>
<img src = "Results/3. Futurism.jpg"><br><br><br>
**Result 1: Futurism Paris**<br><br>
<img src = "Results/4. Paris_Futurism.jpg"><br><br><br>

**2) Scene (New York)**

**Original Image:**<br><br>
<img src = "Results/7. NY.jpg"><br><br><br>
**Art 1: Van Gogh**<br><br>
<img src = "Results/7. Starry Night.jpg"><br><br><br>
**Result 1: New York by Van Gogh**<br><br>
<img src = "Results/7.1. Starry Night at NY.jpg"><br><br><br>

**3) Portrait (Taylor)**

**Original Image:**<br><br>
<img src = "Results/6. Taylor.jpg"><br><br><br>
**Art 1: Art**<br><br>
<img src = "Results/6.1 Art.jpg"><br><br><br>
**Result 1: Taylor portrait**<br><br>
<img src = "Results/6.2 Taylor Art.jpg"><br><br><br>
