Crop image (still updating)
### 
### PROBLEM
A CT scan or computed tomography scan (formerly known as computed axial tomography or CAT scan) is a medical imaging technique used in radiology to get detailed images of the body noninvasively for diagnostic purposes. CT nowadays becomes more and more important for a variety of purposes including damages to bones, injuries to internal organs, problems with blood flow, stroke, and cancer. Thanks to these advantages people can apply AI technology to analyze X-ray images precisely. If people can handle the algorithm,  it will create a fast and economic method to detect sick as well as possible. With the developing of the AI field especially in image processing and the computation of computers, as well as the proficiency in deep learning algorithms, the accuracy of AI applications can reach or surpass people's work.


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

