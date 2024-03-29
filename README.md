
## Bird Species Classification Robustness
This repository contains the project "Enhancing Robustness in Bird Species Classification through Adversarial Training" by Shubhankar Joshi, Chelsi Jain, and Shrirang Patil.

### Overview
The goal of this project is to enhance the robustness of bird species classification models against adversarial attacks. Adversarial attacks involve crafting input images with imperceptible perturbations that lead to incorrect predictions by the model, exposing vulnerabilities in deep learning systems. We focus on three types of adversarial attacks: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and Basic Iterative Method (BIM).

To counter these threats, we employ adversarial training, a defense mechanism that involves training models on adversarial examples to improve their robustness. We utilize three pre-trained models: InceptionV3, EfficientNetB3, and WideResNet50, which are fine-tuned on a dataset of bird species images. The models are then retrained on a combined dataset consisting of both the original and adversarial images to enhance their resilience against the specific attack types.

Our research contributes to the growing body of work on adversarial robustness in deep learning, providing insights into developing more secure and reliable machine learning models for applications in ecological research and conservation.

#### Dataset
The dataset used in this project consists of 84,635 training images across 525 bird species. <br> Dataset link - https://www.kaggle.com/datasets/gpiosenka/100-bird-species

#### License
This project is licensed under the MIT License - see the LICENSE file for details.
