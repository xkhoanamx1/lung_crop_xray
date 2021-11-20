Crop image (still updating)
### 
### PROBLEM
A CT scan or computed tomography scan (formerly known as computed axial tomography or CAT scan) is a medical imaging technique used in radiology to get detailed images of the body noninvasively for diagnostic purposes. CT nowadays becomes more and more important for a variety of purposes including damages to bones, injuries to internal organs, problems with blood flow, stroke, and cancer. Thanks to these advantages people can apply AI technology to analyze X-ray images precisely. If people can handle the algorithm,  it will create a fast and economic method to detect sick as well as possible. With the developing of the AI field especially in image processing and the computation of computers, as well as the proficiency in deep learning algorithms, the accuracy of AI applications can reach or surpass people's work.

The most efficency method obivious is U-Net (architecture Unet was originally designed for grey scale (one channel) biomedical image segmentation from microscopy
data sets). Some experiment has shown the accuracy of U-Net reach over 96 % in validation set (1). Even though this architecture is not compatible with bad scalling X-ray Images (for example: wrong possition when taking picture, patient has lost one part of lung,..). In this case U-net just detect one part or the whole picture as contour.
Due to these disadvantages, we contribute this repotory, which has ability to detect the contour in very bad lung xray image and compute Dice score to assess the method. In practice we had discovered the higest value of the mean in histogram of 1D picture is back-bone possition. To combine with Unet, finially we observe lung coordinate properly.


### UNET
Using U-net for Medical Segmentation.
The paper is from https://arxiv.org/pdf/1505.04597.pdf

![image](https://user-images.githubusercontent.com/33461503/122873769-5e735200-d35c-11eb-9c03-ec3099519c9d.png)


The image-mask dataset was augmented with opencv, pillow and scikit image. The images were augmented by adjusting the brightness, contrast (using opencv), log value (darkness adjustment - using scikit image), whereas the masks were dilated using pillow in order to get more information from the images.

Room for improvements:
* The model will be improved if combining between residual block and Unet. 
* The resblock is inserted in each left-side Unet block to keep boundaries of original features.
* Skip connection at the triangle bottom can be used 3-4 residual blocks.


### Dataset
The training is done with two chest x-rays datasets: [Shenzen & Montogomery CXR dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets). The Montgomery County dataset includes manually segmented lung masks, whereas Shenzhen Hospital dataset was manually segmented by Stirenko et al. 

### Prerequisites
Python 3.6+, Tensorflow 2.0+

## Results

(1)-Article Machine Learning Applications of Convolutional Neural Networks and Unet Architecture to Predict and Classify Demosponge Behavior Dominica Harrison 1,2, Fabio Cabrera De Leo 2,3, Warren J. Gallin 1 , Farin Mir 1 , Simone Marini 4,5 and Sally P. Leys 1,*
