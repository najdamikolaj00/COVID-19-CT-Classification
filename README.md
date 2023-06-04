# Image Classification using Convolutional Neural Networks

## Project Overview

The objective of this project is to develop machine learning approaches for Covid-19 image classification task, with the aim of providing constructive criticism for the data split approach proposed by the authors of the challenge. The project leverages convolutional neural networks (CNNs) to classify images from the provided dataset. In addition to presenting a literature review and implementation details, this project provides a comparison of model performance and a summary of the findings.

## [Dataset & Challenge](https://github.com/UCSD-AI4H/COVID-CT)

The data for the training set were prepared as part of the Challenge - Grand Challenge on COVID-19 diagnosis from CT images. The images of COVID-19 positive were taken from scientific papers from medRxiv and bioRxiv publications, and the COVID-19 negative from sources such as the MedPix database and PubMed Central.

## Experimental Setup

The project is developed in Python programming language and executed in Jupyter Notebooks. PyTorch and scikit-learn are used for the development of CNNs.

The project uses two different dataset split approaches to train and test the models. The best model presented in the authors' approach is tested, followed by other proposed CNNs. The performance of the models is compared using common metrics, including F1-score, AUC and Accuracy.

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
|   |-- results/(models' performance)
|   |-- data_loader.py
|   |-- densenet169_test.py
|   |-- enhancedcnn_test.py
|   |-- enhancedcnn.py
|   |-- results_visualization.ipynb
|   |-- simplecnn_test.py
|   |-- simplecnn.py
|-- Classification_our_approach/
|   |-- results/(models' performance)
|   |-- CT_COVID.txt
|   |-- CT_NonCOVID.txt
|   |-- data_loader.py
|   |-- densenet169_test.py
|   |-- enhancedcnn_test.py
|   |-- enhancedcnn.py
|   |-- results_visualization.ipynb
|   |-- simplecnn_test.py
|   |-- simplecnn.py
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
|-- COVID_19_CT_Classification_UNI_Project.pdf
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
3. ... Go on!

## Conclusion
  In this project, we have explored the topic of workflow optimization in the context of machine learning models. The main objective was to compare the performance of three different models (DenseNet169, SimpleCNN, and EnhancedCNN) across two different workflows (Workflow A and Workflow B). The evaluation of these models was based on three metrics: F1-score, Accuracy, and AUC.
  Initially, we preprocessed the dataset and split it into training and validation sets using k-fold cross-validation. Then, we trained the models using a fixed number of epochs and recorded their performance on the training and validation set for each workflow which is shown on the learning curves plots.
  The analysis of the results revealed interesting insights. For Workflow A, the T-test and Wilcoxon test showed significant differences in performance between the models for the F1-score, Accuracy, and AUC metrics. In particular, DenseNet169 exhibited superior performance compared to other models in terms of the F1-score and AUC. For Workflow B, SimpleCNN indicated better performance in classification. This suggests that the choice of workflow has a substantial impact on the models' performance in terms of these metrics.
  Overall, the statistical analysis highlights the importance of model selection in proposed workflows. The results demonstrate the significant differences in performance among the three models and emphasize the need for careful consideration when choosing the most suitable model for a given workflow.
