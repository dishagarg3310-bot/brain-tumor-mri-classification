\# Brain Tumor MRI Image Classification



A deep learning project that classifies brain MRI scans into four classes: \*\*glioma, meningioma, pituitary tumor and no tumor\*\*. It includes a custom CNN built from scratch, MobileNetV2 transfer learning, model evaluation and comparison, Grad-CAM explainability and a Streamlit web app for real-time predictions.



> \*\*Disclaimer:\*\* This is an educational prototype, not a medical device. It must not be used for diagnosis.



\## Results



| Model | Test accuracy | Notes |

|---|---|---|

| \*\*Custom CNN (final model)\*\* | \*\*82.52%\*\* | Macro F1 0.82 |

| MobileNetV2 (frozen base) | 80.49% | Higher validation accuracy, lower on test |

| MobileNetV2 (fine-tuned) | XX.XX% | Overfitted on this small dataset |



Per-class results of the custom CNN on the test set:



| Class | Precision | Recall | F1-score |

|---|---|---|---|

| Glioma | 0.94 | 0.85 | 0.89 |

| Meningioma | 0.73 | 0.57 | 0.64 |

| No tumor | 0.74 | 0.94 | 0.83 |

| Pituitary | 0.84 | 0.98 | 0.91 |



\## Workflow



1\. Dataset exploration: class balance, image sizes, duplicate and corrupt file checks, brightness and contrast analysis, hypothesis testing

2\. Preprocessing: resize to 224 x 224, pixel scaling to 0-1

3\. Augmentation (training set only): rotation, shifts, flips, zoom, brightness changes

4\. Models: custom CNN (4 convolution blocks with batch normalization and dropout) and MobileNetV2 with ImageNet weights (frozen, then fine-tuned)

5\. Training with early stopping, model checkpointing and learning-rate scheduling

6\. Evaluation: accuracy, precision, recall, F1-score, confusion matrix and training curves

7\. Explainability with Grad-CAM

8\. Deployment as a Streamlit app



\## Repository structure



brain-tumor-mri/

├── app.py # Streamlit application

├── Brain\_Tumor\_MRI\_Classification.ipynb # Full analysis, training and evaluation notebook

├── requirements.txt

├── models/

│ ├── brain\_tumor\_final\_model.h5 # Final model used by the app (custom CNN)

│ ├── custom\_cnn\_checkpoint.h5

│ ├── transfer\_model\_checkpoint.h5

│ ├── finetuned\_model.h5

│ └── class\_names.json

└── README.md





\## Run the app locally



```bash

pip install -r requirements.txt

streamlit run app.py

```



The app opens at http://localhost:8501. Upload a brain MRI (JPG or PNG) to see the predicted class and the confidence for each class.



\## Dataset



\[Labeled MRI Brain Tumor Dataset](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset) from Roboflow (author: ali-rostami), licensed under CC BY 4.0. It contains 2,443 images split into 1,695 training, 502 validation and 246 test images. The dataset is not included in this repository; download it separately to re-run the notebook.



\## Limitations



\- Meningioma is the weakest class (recall 0.57), so many meningioma scans are missed.

\- The app always returns one of the four classes, even for an image that is not a brain MRI.

\- The model was trained on a single dataset and has not been validated on scans from other hospitals or scanners.

\- It is a decision-support prototype and every prediction must be reviewed by a qualified radiologist.



\## Author



Disha

