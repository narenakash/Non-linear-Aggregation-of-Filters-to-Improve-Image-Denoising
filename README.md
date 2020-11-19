# Digital Image Processing Project: Non-linear Aggregation of Filters to Improve Image Denoising

The repository contains the implementatiion of the paper [Non-linear Aggregation of Filters to Improve Image Denoising](https://arxiv.org/pdf/1904.00865.pdf). This project is a partial requirement of the Digital Image Processing course at [IIIT Hyderabad](https://www.iiit.ac.in/) instructed by [Prof. Ravi Kiran Sarvadevabhatla](https://ravika.github.io/) in the Monsoon 2020 semester. 

## Paper Overview

The paper introduces a novel aggregation method to efficiently perform image denoising. Preliminary filters are aggregated in a non-linear fashion, using a new metric of pixel proximity based on how the pool of filters reaches a consensus. It provides a theoretical bound to support our aggregation scheme, its numerical performance is illustrated and we show that the aggregate significantly outperforms each of the preliminary filters.

## To Run The Demo

#### 1. Clone the Repository

```bash
git clone https://github.com/Digital-Image-Processing-IIITH/project-revision.git
```

#### 2. Setting Up Virtual Environment
```bash
conda create --name envname python=3.8
conda activate envname
```
Ensure that you install all the dependencies in the virtual environment before running the program. We have used `Python 3.8` during the development process. Do ensure that you have the same version before running the code.

#### 3. Running On Local Machine
```bash
cd src
python3 main.py
```
Do note that the training process may take several hours. The team members used the Ada High Performance Cluster of IIIT Hyderabad for training the model.

#### Pretrained Model 
The link to the best model obtained upon training is [here](https://drive.google.com/drive/folders/1TrwRDwMP2HgtHFfrVahM5bbug3ctNWaF).

To use the pretrained model download the pkl file to `src/` and in `src/main.py` set:
```
loadModel = True
```

## Dependencies
The following command will install the packages according to the configuration file `src/req.txt`.
```bash
pip3 install -r src/req.txt
```
- cycler==0.10.0
- joblib==0.17.0
- kiwisolver==1.3.1
- matplotlib==3.3.3
- numpy==1.19.4
- pandas==1.1.4
- Pillow==8.0.1
- pycobra==0.2.3
- pyparsing==2.4.7
- python-dateutil==2.8.1
- pytz==2020.4
- scikit-learn==0.23.2
- scipy==1.5.4
- seaborn==0.11.0
- six==1.15.0
- threadpoolctl==2.1.0

## Repository Organization

#### Repository Structure

```
project-revision
├── dataset
|  ├── test
|  ├── train
├── docs
├── misc
|   ├── dataset
|   ├── reszie.py
├── src
|   ├── denoise
|   |   |── __init__.py
|   |   |── denoise.py
|   |   |── errors.py
|   ├── noise
|   |   |── __init__.py
|   |   |── errors.py
|   |   |── noise.py
|   ├── results
|   ├── cobra.py
|   ├── cobramachine.py
|   ├── denoise.py
|   ├── helper.py
|   ├── main.py
|   ├── req.txt
├── README.md
├── guidelines.md
├── proposal.md
```

The `dataset` folder contains the train and test splits of the dataset. The `docs` folder contains the project proposal documents, and project presentation slide-decks.The `results` folder holds the results of running the implementation on the images. The `src` folder contains the source code and the dependent libraries. 

The `helper.py` has the denoising evaluation function and the training data loader function. The `cobra.py` and `cobramachine.py` files have the COBRA algorithm defined in the Python library `pycobra`. `denoise/denoise.py` and `noise/noise.py` files contains all the denoising algoithms and noise models respectively. `main.py` is the starting point of the software execution. It calls all the required functions in the appropriate order and stores the final model.  

#### Dataset Creation

The dataset is made of up 26 images taken from the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) public benchmark dataset for face verification. The images were resized to `64 x 64`. The implementation considers 25 images in the training set, and takes one image for testing purpose. 

#### High-level Working of Code
The program takes 25 input noise-free images and adds seven standard noise models to it on the run-time. The noisy images are run through seven classical denoising algorithms. We train the COBRA machine using the noisy and denoised images to perform non-linear aggregation of the preliminary filters. The best model obtained from the training process is used for image denoising. 

## Team Members

- [Amogh Tiwari](https://researchweb.iiit.ac.in/~amogh.tiwari/)
- [Dolton Fernandes](https://doltonfernandes.github.io/)
- [George Tom](https://georg3tom.github.io/)
- [Naren Akash R J](https://researchweb.iiit.ac.in/~naren.akash/)

All the team members are undergraduate research students at the [Center for Visual Information Technology, IIIT Hyderabad](http://cvit.iiit.ac.in/), India.

## Acknowlegements
The team members are grateful towards the instructor Prof. Ravi Kiran Sarvadevabhatla and mentor teaching assistant Soumyasis Gun. They also acknowledge [CVIT, IIIT Hyderabad](http://cvit.iiit.ac.in/) for providing access to the Ada High Performance Cluster for training the machine learning model. 

## Licence and Citation
The software can only be used for personal/research/non-commercial purposes. To cite the original paper:
```
@InProceedings{10.1007/978-3-030-52246-9_22,
author="Guedj, Benjamin and Rengot, Juliette",
editor="Arai, Kohei and Kapoor, Supriya and Bhatia, Rahul",
title="Non-linear Aggregation of Filters to Improve Image Denoising",
booktitle="Intelligent Computing",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="314--327",
abstract="We introduce a novel aggregation method to efficiently perform image denoising. Preliminary filters are aggregated in a non-linear fashion, using a new metric of pixel proximity based on how the pool of filters reaches a consensus. We provide a theoretical bound to support our aggregation scheme, its numerical performance is illustrated and we show that the aggregate significantly outperforms each of the preliminary filters.",
isbn="978-3-030-52246-9"
}
```
