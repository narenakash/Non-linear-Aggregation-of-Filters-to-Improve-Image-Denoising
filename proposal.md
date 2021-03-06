# Project Proposal
- Project ID: 22
- Project Title: Non-linear aggregation of filters to improveimage denoising
- Project Members:  
    + Naren Akash R J
    + George Tom
    + Amogh Tiwari
    + Dolton Fernandes
- Github link: [repo](https://github.com/Digital-Image-Processing-IIITH/project-revision)

## Problem Definition
A common problem encountered during image acquisition is of **noise** creeping in to images due to various reasons like poor lighting conditions, sensor issues, approximations during digitalization, image transmission etc. Therefore, *image denoising* is necessary to improve the quality of the image.

Many popular *denoising* techniques exist nowadays, but *image smoothening* during denoising is still a problem. Each denoising method has its pros and cons. We can use a combination of denoising techniques to denoise the image, and it needs to be adapted to different types of noise in the image.

## Main goal(s) of the project
+ Make modules to add artificial noise to images (Gaussian, Poisson, salt-and-pepper, speckle, random suppression, multi, etc.)
+ Code classical algorithms to denoise images (Gaussian filter, Median filter, Bilateral filter, non-local means, TV-Chambolle, Richardson-Lucy deconvolution, inpainting, etc.)
+ Use COBRA module and implement an image denoising algorithm.
+ Implement feature extraction from images.
+ Code to evaluate the denoising quality i.e. implement loss functions like Root Mean Squared Error (RMSE), Peak Signal-to-Noise Ratio (PSNR).
+ Training and optimizing the parameters.
+ Gather images for the dataset.

## Expected Results
COBRA denoising method enhances the performance of the preliminary filters. We will use an aggregation of multiple filters, which allows us to take advantage of the abilities of all the individual filters which in turn, can adapt better to unknown noise levels. As our final result, we will demonstrate that such a non-linear aggregation of filters significantly outperform each of the individual filters. 

## Project Timeline
+ 19 Oct - 25 Oct:
    + Read and understand the given paper and all related papers
    + Set up the environment: Install all necessary libraries
+ 26 Oct - 30 Oct:
    + Use author’s set of images to run our code
    + Implement a basic denoising code
+ **Mid Evaluation: 31 October**
+ 01 Nov - 05 Nov: 
    + Further improve the denoising code
    + Implement a script to compare the similarity of 2 images
+ 06 Nov - 10 Nov:
    + Compare results of our code with other denoising methods
    + Compare results of our code with results from the paper
    + Do the above 2 steps for our own dataset of images
+ 11 Nov - 18 Nov: 
    + Perform a final set of tests and compare results. Make any improvements if needed
    + Prepare Presentation
    + Reserve time: To incorporate for delays
+ **Final Evaluation: 19-25 Nov**


## Is there a dataset you need ? How do you plan to get it?
We need a set of 25 images for training. The authors of the paper have provided a set of images. However, if time permits, as an additional challenge we plan to create a database by accumulating suitable images from the internet and see how our algorithm performs with a new set of training images.
