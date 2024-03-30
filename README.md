
## Bird Species Classification Robustness
This repository contains the project "Enhancing Robustness in Bird Species Classification through Adversarial Training" by Shubhankar Joshi, Chelsi Jain, and Shrirang Patil.

### Overview
The goal of this project is to enhance the robustness of bird species classification models against adversarial attacks. Adversarial attacks involve crafting input images with imperceptible perturbations that lead to incorrect predictions by the model, exposing vulnerabilities in deep learning systems. We focus on three types of adversarial attacks: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and Basic Iterative Method (BIM).

To counter these threats, we employ adversarial training, a defense mechanism that involves training models on adversarial examples to improve their robustness. We utilize three pre-trained models: InceptionV3, EfficientNetB3, and WideResNet50, which are fine-tuned on a dataset of bird species images. The models are then retrained on a combined dataset consisting of both the original and adversarial images to enhance their resilience against the specific attack types.

Our research contributes to the growing body of work on adversarial robustness in deep learning, providing insights into developing more secure and reliable machine learning models for applications in ecological research and conservation.

#### Dataset
The dataset used in this project consists of 84,635 training images across 525 bird species. <br> Dataset link - https://www.kaggle.com/datasets/gpiosenka/100-bird-species<br>
The project is live at https://huggingface.co/spaces/shubhu163/BirdClassRobustness

<p align="left">
  <img src="https://github.com/shubhu163/BirdClass_Robustness/assets/71623089/f0fd258e-5822-4227-aded-9f1d45534037" width="300" alt="Bird Species Sample 1">
  <img src="https://github.com/shubhu163/BirdClass_Robustness/assets/71623089/3bbbed16-575e-4515-9b45-6696a1e09ed6" width="300" alt="Bird Species Sample 2">
  <br>
  <em>Original Dataset</em> &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp;   &nbsp;  &nbsp;  &nbsp; &nbsp;   &nbsp;  &nbsp;  &nbsp;  &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp; <em>Adversarial Dataset</em>
</p>


