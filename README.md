# Image Classification using Convolutional Neural Networks

## Project Overview

The objective of this project is to develop and compare state-of-the-art machine learning approaches for Covid-19 image classification tasks, with the aim of providing constructive criticism for the data split approach proposed by the authors of the challenge. The project leverages convolutional neural networks (CNNs) and other machine learning algorithms to classify images from the provided dataset. In addition to presenting a literature review and implementation details, this project provides a comparison of model performance and a summary of the findings.


## [Dataset & Challenge](https://github.com/UCSD-AI4H/COVID-CT)

The data for the training set were prepared as part of the Challenge - Grand Challenge on COVID-19 diagnosis from CT images. The images of COVID-19 positive were taken from scientific papers from medRxiv and bioRxiv publications, and the COVID-19 negative from sources such as the MedPix database and PubMed Central.

## Experimental Setup

The project is developed in Python programming language and executed in Jupyter Notebooks on Google Colab. PyTorch and scikit-learn are used for the development of CNNs and other machine learning models.

The project uses four different dataset split approaches to train and test the models. The best model presented in the authors' approach is tested, followed by other state-of-the-art methods and the proposed CNN. The performance of the models is compared using common metrics, including F1-score, confusion matrix, and ROC Curve.

## Results

## Repository Structure
```
|-- Classification_Authors_Based_Data_Split/
|   |-- COVID/
|   |   |-- testCT_COVID.txt
|   |   |-- trainCT_COVID.txt
|   |   |-- valCT_COVID.txt
|   |-- NonCovid/
|   |   |-- testCT_COVID.txt
|   |   |-- trainCT_COVID.txt
|   |   |-- valCT_COVID.txt
|   |-- data_loader.py
|-- Classification_our_approach/
|-- Data/
|   |-- CT_COVID/
|   |   |-- images
|   |-- CT_NonCOVID/
|   |   |-- images
|-- LaTeX_script/
|   |-- imagesDatasetSection/
|   |   |-- images
|   |-- llncs.cls
|   |-- MainPaper.tex
|   |-- references.tex
|   |-- splncs04.bst
|-- .gitignore
|-- README.md
|-- requirements.txt
```
## Usage

To run the project, follow these steps:

1. Clone the repository:
```
git clone https://github.com/najdamikolaj00/COVID-19-CT-Classification.git
```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. ...

## Conclusion
