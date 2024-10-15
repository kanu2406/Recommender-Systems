# Collaborative Filtering Recommender System

**Team:** PSG-IASD  
**Members:**  
- Kanupriya Jain  
- Othman Hicheur  
- Nan An  

## Project Overview
This repository contains the code and analysis for building a collaborative filtering-based recommender system which was done as a part of project for Data Science Lab. The system predicts user ratings for movies using various machine learning techniques, including matrix factorization and deep learning approaches.

## Methods Implemented
1. **Alternating Least Squares (ALS) Matrix Factorization**
   - Decomposes the user-item interaction matrix into lower-dimensional matrices.
   - **Location:** `Code3_Alternating_Least_Squares(ALS).ipynb`
   - **Best RMSE:** 3.470

2. **Probabilistic Matrix Factorization (PMF)**
   - A Bayesian learning-based approach for parameter estimation using low-rank matrices.
   - **Location:** `Code2_Probabilistic_Matrix_Factorization(PMF).ipynb`
   - **Best RMSE:** 1.035

3. **Neural Collaborative Filtering (NCF)**
   - Deep learning model that uses embeddings to represent users and items, and neural networks to predict interactions.
   - **Location:** `Code1_Neural_Collaborative_Filtering_Model.ipynb`
   - **Best RMSE:** 0.879

## Preprocessing
- The movie genre feature was one-hot encoded and incorporated into the input features.
- **Preprocessing Steps:** Adding genres as features, users and items embedding , and genres.

## Post-processing
- Applied strategies to map continuous predictions to the discrete rating scale (0.5 to 5.0).
- Methods include rounding, classification, and custom mapping to the rating scale.


4. Run the following notebooks:
   - **ALS Matrix Factorization:** `Code3_Alternating_Least_Squares(ALS).ipynb`
   - **Probabilistic Matrix Factorization:** `Code2_Probabilistic_Matrix_Factorization(PMF).ipynb`
   - **Neural Collaborative Filtering (NCF):** `Code1_Neural_Collaborative_Filtering_Model.ipynb`

## Results
- **ALS Matrix Factorization:**
  - RMSE: 3.470
- **Probabilistic Matrix Factorization (PMF):**
  - RMSE: 1.035
- **Neural Collaborative Filtering (NCF):**
  - RMSE: 0.879




