still updating...
### 
## PROBLEM
A CT scan or computed tomography scan (formerly known as computed axial tomography or CAT scan) is a medical imaging technique used in radiology to get detailed images of the body noninvasively for diagnostic purposes. CT nowadays becomes more and more important for a variety of purposes including damages to bones, injuries to internal organs, problems with blood flow, stroke, and cancer. Thanks to these advantages people can apply AI technology to analyze X-ray images precisely. If people can handle the algorithm,  it will create a fast and economic method to detect sick as well as possible. With the developing of the AI field especially in image processing and the computation of computers, as well as the proficiency in deep learning algorithms, the accuracy of AI applications can reach or surpass people's work.

The most efficency method obivious is U-Net (architecture Unet was originally designed for grey scale (one channel) biomedical image segmentation from microscopy
data sets). Some experiment has shown the accuracy of U-Net reach over 96 % in validation set (1). Even though this architecture is not compatible with bad scalling X-ray Images (for example: wrong possition when taking picture, patient has lost one part of lung,..). In this case U-net just detect one part or the whole picture as contour.

U-Net works very well on the stage of lung part detection, but sometimes it fails in case of very bad lung X-ray picture (the wrong position when taking pictures, or the contrast is not good...). So we need more relatively root point combine with U-net to enhance the accuracy of this method, that is Back-bone position.

Due to these disadvantages, we contribute this repotory, which has ability to detect the contour in very bad lung xray image and compute Dice score to assess the method. In practice we had discovered the higest value of the mean in histogram of 1D picture is back-bone possition. To combine with Unet, finially we observe lung coordinate properly.
### GENERAL ALGORITHM
![Lung-segmentation-using-U-Net-before-training-the-convolutional-neural-network-a-the](https://user-images.githubusercontent.com/45206333/142722675-12e401d1-522c-4fa0-a111-2f057dba0f6f.png)

Because X-ray image is not suitable with AI applications so:
* In the first step, we convert the Xray image with a high density of pixel(DICOM file) to .png 256x256
* After that images are fed into U-net and stored all predictions in the new folder.
* Following that we calculate the mean of histograms to detect back-bone.
* The final step is that position of the back-bone combine with predictions to have a new lung coordinate.

### UNET
Using U-net for Medical Segmentation.
The paper is from [here](https://arxiv.org/pdf/1505.04597.pdf)

![image](https://user-images.githubusercontent.com/33461503/122873769-5e735200-d35c-11eb-9c03-ec3099519c9d.png)


The image-mask dataset was augmented with opencv, pillow and scikit image. The images were augmented by adjusting the brightness, contrast (using opencv), log value (darkness adjustment - using scikit image), whereas the masks were dilated using pillow in order to get more information from the images.

Room for improvements:
* The model will be improved if combining between residual block and Unet. 
* The resblock is inserted in each left-side Unet block to keep boundaries of original features.
* Skip connection at the triangle bottom can be used 3-4 residual blocks.

### Back-bone position

By archiving stunning result in back-bone position, we calculate histogram of x-axis following y-axis (1:256), the highest score (the lightest part in lung X-ray grayscale picture is the back-bone position)


![grapth](https://user-images.githubusercontent.com/45206333/142733609-ea0ab76b-f8dc-442f-99e3-7bb574b0898a.png)


### Dataset
* The training is done with two chest x-rays datasets: [Shenzen & Montogomery CXR dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets). The Montgomery County dataset includes manually segmented lung masks, whereas Shenzhen Hospital dataset was manually segmented by Stirenko et al. 
* SIIM-FISABIO-RSNA COVID-19 Kaggle.

## Prerequisites
Python 3.6+, Tensorflow 2.0+

## Results
One example of good result:
![grapth2](https://user-images.githubusercontent.com/45206333/142734031-7529f01f-6a61-4211-be72-e378c8922ea7.png)


* Wrong 8/1263 pictures of SIIM-FISABIO-RSNA COVID-19 Kaggle test data  
* Dice coefficient can reach up to 93-94% with our training model (unet with resblock). With test set, the prediction can include noises outside lung region or only one side of lung (picture above). Our cropping algorithm can fix it.
* We set crop command at the end of this application, if it is not suitable with your demand you can extract the coordinates(xmin, ymin, xmax, ymax) to define lung positon.

## References
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
* (1)-Article Machine Learning Applications of Convolutional Neural Networks and Unet Architecture to Predict and Classify Demosponge Behavior Dominica Harrison 1,2, Fabio Cabrera De Leo 2,3, Warren J. Gallin 1 , Farin Mir 1 , Simone Marini 4,5 and Sally P. Leys 1,*
