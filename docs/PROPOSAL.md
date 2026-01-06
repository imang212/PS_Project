# Project Proposal

## Topic Name in Czech:
**Systém pro analýzu, klasifikaci a třídění objektů v reálném čase**

## Topic Name in English:
**Real-time Object Analysis, Classification, and Sorting System**

### Topic Annotation
#### Topic Motivation:
Automated quality control and product sorting is crucial in many industrial and agricultural sectors. This thesis focuses on the design and implementation of a system capable of analyzing video signals in real-time, recognizing, segmenting, and classifying objects, and physically sorting them based on this classification. An example could be a potato or apple sorting machine where defective pieces need to be separated. The goal is to create a prototype running on accessible hardware, such as a Raspberry Pi with an AI accelerator, and thus demonstrate a practical application of modern computer vision and machine learning methods.

#### Thesis Objective:
The objective of this bachelor's thesis is to **design, implement, and test a system for video signal analysis, object classification, and subsequent hardware-based sorting**. The system will be built on the Raspberry Pi platform with an AI module and will be capable of processing images in real-time, collecting data about classified objects, visualizing them, and controlling external hardware for physical sorting.

#### Thesis Goals (Detailed Breakdown):
*   **Research:** Overview of methods for object recognition, segmentation, and classification in images. Analysis of available hardware solutions for AI on embedded systems (e.g., AI HAT for Raspberry Pi).
*   **System Design:** System architecture including both software and hardware components. Definition of communication protocols between individual parts.
*   **Data Collection and Preparation:** Creating or obtaining a dataset for model training (e.g., images of apples/potatoes of various qualities).
*   **Model Implementation:** Selection, training, and optimization of an object classification model for deployment on Raspberry Pi.
*   **Software Development:** Implementation of software for:
    *   1. Video signal analysis from the camera.
    *   2. Facilitating data collection about detected objects (category, detection time).
    *   3. Data visualization (e.g., in a simple web interface).
*   **Hardware Integration:** Connection to a hardware system (e.g., servo, lever) that performs physical sorting of classified objects.
*   **Testing and Evaluation:** Testing the entire system in a real scenario, evaluating classification accuracy and sorting reliability.

#### Thesis Outputs:
The output of the thesis will be a **functional device prototype** capable of analyzing images in real-time, classifying objects, and sorting them accordingly. It will also include a **software application** for data collection and visualization, and **documentation** describing the design, implementation, and testing results.

# Tasks:
Create software for video signal analysis.