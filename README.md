# [COVID Detection using X-ray Images](https://github.com/pateldivyam26/COVID-Detection-using-X-Ray-Images)

## Background
The outbreak of Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-COV-2) has caused more than 17.5 million cases of Corona Virus Disease (COVID-19) in the world so far, with that number continuing to grow. To control the spread of the disease, screening large numbers of suspected cases for appropriate quarantine and treatment is a priority.

The standard COVID-19 tests are called PCR (Polymerase chain reaction) tests, but they have certain limitations such as being time-consuming and yielding significant false-negative results. This project aims to explore the use of Artificial Intelligence and Machine Learning techniques to develop a parallel diagnosis/testing procedure for COVID-19 using X-ray images.

## Dataset Used
The dataset used for this project is available on [Kaggle: COVID Detection Dataset](https://www.kaggle.com/competitions/stat946winter2021/overview).

## Technologies Used
The following technologies were used in this project:
- [Scikit Learn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/)
- [PyTorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/stable/index.html)
- [Seaborn](https://seaborn.pydata.org/)
- [OpenCY](https://opencv.org/)

## Pipeline
1. **Data Preparation and Preprocessing:**
   - The grayscale X-ray images were resized to (128, 128) to ensure consistent data representation.
   - Principal Component Analysis (PCA) was applied to reduce the dimensionality of the feature vectors.
   - Linear Discriminant Analysis (LDA) was used to increase the separability between the COVID and non-COVID classes.
   - Heatmap generation provided insights for further analysis.

<p  align="center">
<img src="https://github.com/pateldivyam26/COVID-Detection-using-X-Ray-Images/assets/79200448/4669f04e-10be-4310-9c64-081c48b1444a" alt="form"><br>
<i>Fig. 1: Chest X-Ray Image</i>
</p>
<p  align="center">
<img src="https://github.com/pateldivyam26/COVID-Detection-using-X-Ray-Images/assets/79200448/fc1eba9a-c581-41aa-bbc0-0531efbb426f" alt="form">
<img src="https://github.com/pateldivyam26/COVID-Detection-using-X-Ray-Images/assets/79200448/a06e1131-a396-436b-a1c8-34e27f016180" alt="form"><br>
<i>Fig. 2: Heatmap of Images</i>
</p>

2. **Machine Learning Models:**
   - `Random Forest:` Achieved an accuracy of 86.2%.
   - `Bayes Classification- Gaussian:` Achieved an accuracy of 69.1%.
   - `Light Gradient Boosting (LGBM):` Achieved the highest accuracy of 88%.
   - `XGBoost:` Achieved an accuracy of 86.2%.
   - `Logistic Regression:` Achieved an accuracy of 78%.
   - `Support Vector Classifier (SVC):` Achieved an accuracy of 86.7%.
   - `K-Nearest Neighbors (KNN):` Achieved an accuracy of 85.3%.
   - `Multi-layer Perceptron (MLP):` Achieved an accuracy of 81.5% with PCA.

3. **Evaluation of Models:**
   - The models were evaluated using precision, recall, F1 score, and accuracy scores.

4. **Deep Learning Models:**
   - `CNN Scratch 1:` Achieved a validation accuracy of 92% and testing accuracy of 93%.
   - `CNN Scratch 2:` Achieved a validation accuracy of 91% and testing accuracy of 90%.
   - `ResNet50:` Achieved a validation accuracy of 90.62% and testing accuracy of 87%.
   - `VGG16:` Achieved a validation accuracy of 87% and testing accuracy of 85%.

For more details, please refer to the [project report](https://github.com/pateldivyam26/COVID-Detection-using-X-Ray-Images/blob/main/B20EE082_B20AI014_REPORT.pdf) included in this repository.

## Deployment
<p  align="center">
<img src="https://github.com/pateldivyam26/COVID-Detection-using-X-Ray-Images/assets/79200448/68cf6428-785f-49cd-a9b4-ef07f313b41a" width = "960px" alt="form"><br>
<i>Fig. 3: Home Page</i>
</p>
<p  align="center">
<img src="https://github.com/pateldivyam26/COVID-Detection-using-X-Ray-Images/assets/79200448/ef7c1752-0e94-4bd6-b105-c2eafae49ae9" width = "960px" alt="form"><br>
<i>Fig. 4: Result Page</i>
</p>

## Team

| Name                                            | Roll Number |
| ----------------------------------------------- | ----------- |
| [Divyam Patel](https://github.com/pateldivyam26) | B20EE082   | 
| [Jaimin Gajjar](https://github.com/jaimin001)    | B20AI014    |
