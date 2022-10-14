# Automated Detection of Cardiomegaly from Chest X-Ray Images 

### Background: Pattern recognition of different diseases from medical imaging studies using deep learning is evolving rapidly, with some algorithms performing better than expert radiologists in identifying these diseases. One area where deep learning algorithms could improve clinical workflows in a variety of medical settings is in automated cardiomegaly detection from chest X-ray images. Therefore, we developed and evaluated a series of deep learning algorithms for the classification of cardiomegaly from chest X-ray images. 

<hr />

### Methods: A subset of patients from the NIH Chest X-ray dataset consisting of positive (cardiomegaly) and negative (no cardiomegaly) samples were used to develop and validate deep learning classification algorithms. After image preprocessing, a variety of models with different architectures and parameters were constructed, evaluated, and compared. Model performance was predominantly measured via the area under the receiver operating characteristic curve (AUC), but sensitivity, specificity, F1 score, and Matthews correlation coefficient were also examined. Model interpretability was investigated using Grad-CAM. 

<hr />

### Results: Using independent training (N=21386) and test (N=600) sets, seven different deep learning models were developed and compared. Most of the models performed well in detecting cardiomegaly. The best model had an AUC of 0.905 in predicting cardiomegaly. Grad-CAM revealed models tended to focus on areas of the image containing cardiac tissue when classifying a case as having cardiomegaly. 

<hr />
