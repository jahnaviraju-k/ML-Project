#  Audio Feature Analysis for Music Genre Classification

## Project Overview

This project explores the use of machine learning and deep learning algorithms to classify music tracks into genres based on their audio features. Using the **Spotify Tracks Dataset**, which spans **125 music genres**, this study addresses the challenge of genre classification — a complex task due to the subjective and overlapping nature of musical categories.

The ultimate goal is to understand how well audio features can predict genre, and what modeling and preprocessing strategies are most effective when dealing with high cardinality and imbalanced class distributions.

---

## Work on the Project

**In this Project** I handled every aspect of the workflow, from data preprocessing to model training and evaluation. Specifically, I:

- Cleaned and standardized the Spotify dataset for modeling
- Conducted **feature selection and transformation** to improve signal clarity
- Built and evaluated multiple classification models:
  - **Random Forest**
  - **Gradient Boosting (XGBoost)**
  - **Deep Neural Networks (DNN)**
- Analyzed performance challenges related to **class imbalance** and **genre overlap**
- Suggested future research directions for improving genre classification precision

---

##  Project Goals

- Apply supervised learning to classify songs based on audio features
- Evaluate model performance on a **multi-class**, **imbalanced** dataset
- Understand the limitations of traditional models in high-genre overlap scenarios
- Explore how accurate genre classification can enhance user experience in music recommendation systems

---


## Key Findings

**Modest Accuracy Due to Genre Overlap**
- Despite using powerful models, classification accuracy remained modest (e.g., ~60–65%), primarily because many genres had overlapping audio features.
- Used Scikit-learn to train baseline models (e.g., Random Forest and Gradient Boosting Classifier).
- Implemented a confusion matrix to visualize how frequently genres were misclassified as similar genres.
- Plotted confusion matrix heatmaps using Seaborn to understand genre clusters.

**Severe Class Imbalance Affected Model Performance**
- The dataset contained over 125 genres, but most tracks belonged to only a few dominant genres. This imbalance skewed model learning.
- Used value_counts() in Pandas to observe class distribution.
- Implemented Stratified Train-Test Split to preserve minority classes.
- Attempted class weighting in Scikit-learn and class_weight='balanced' in Keras models to mitigate the imbalance.

**Tree-Based Models Outperformed DNNs for Small-Class Problems**
- Tree-based models like Random Forest and XGBoost consistently outperformed deep neural networks in precision, recall, and interpretability.
- Trained models using Scikit-learn’s RandomForestClassifier and XGBoostClassifier.
- Compared performance using metrics like accuracy, macro F1-score, and precision/recall curves.
- Used feature_importances_ from tree models to rank top audio features.

**Deep Neural Networks Struggled with Overfitting**
- The DNN performed well on the training set but poorly on the validation set, indicating overfitting due to high class count and lack of per-class samples.
- Built a DNN using Keras with TensorFlow backend, using dense layers with ReLU activation and softmax output.
- Used early stopping and dropout layers to reduce overfitting.
- Tracked training vs. validation loss with Matplotlib plots to visually confirm divergence.

**Top Features Influencing Genre Classification**
- Audio features like tempo, danceability, instrumentalness, energy, and valence were the most influential in genre prediction.
- Used Random Forest’s feature importance scores to rank influential variables.
- Visualized feature distributions across genres using Seaborn violin plots and histograms.
- Supplemented analysis with correlation heatmaps to detect multicollinearity among features.


