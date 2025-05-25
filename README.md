# 🎬 ML Movie Recommendation & Rating Prediction System

This repository contains the final submission for the **Akbank Global AI Hub Machine Learning Bootcamp - May 2025** project. The project demonstrates supervised learning techniques applied to a real-world movie dataset, focusing on both **movie rating prediction** and **recommendation systems**.

---

## 🔗 Kaggle Notebook Links

- [Movie Recommendation System - Collaborative Filtering (KNN)](https://www.kaggle.com/YOUR-NOTEBOOK-LINK)
- [Movie Rating Prediction - Regression Model](https://www.kaggle.com/YOUR-NOTEBOOK-LINK)
- [Movie Recommendation System - TF-IDF + Cosine Similarity](https://www.kaggle.com/YOUR-NOTEBOOK-LINK)

---

## 📌 Project Overview

This project tackles two major problems using the TMDB movie metadata dataset:

1. **Movie Recommendation Problem** — Recommending similar movies based on user preference or content metadata.
2. **Movie Rating Prediction Problem** — Predicting a movie's IMDb rating using its metadata.

Through this study, I developed three separate models, each documented in its respective Jupyter notebook.

---

## 📁 Notebooks Description

### 1. `GAIH_ML_Bootcamp_May2025_MLMovieRecoSystemProject.ipynb`  
**Technique Used**: Collaborative Filtering using KNN (k-Nearest Neighbors)

**Goal**: Recommending movies that are similar to a user’s selected movie ("Iron Man").

**Result**:  
Recommended movies for *Iron Man*:

1. The Avengers  
2. Avatar  
3. Inception  
4. Star Wars  
5. The Matrix  
6. Iron Man 2  
7. Iron Man 3  
8. Captain America: The First Avenger  
9. The Empire Strikes Back  
10. Avengers: Age of Ultron  

> This model was built using `NearestNeighbors` from Scikit-learn with cosine distance metric on a user-item matrix.

---

### 2. `GAIH_ML_Bootcamp_May2025_MLMoviePredProject.ipynb`  
**Technique Used**: Linear Regression (Supervised Learning)

**Goal**: Predicting the IMDb rating of a movie based on its metadata (vote count, genres, popularity, etc.)

**Result**:
| Actual Rating | Predicted Rating |
|---------------|------------------|
| 5.20          | 5.32             |
| 5.70          | 5.24             |
| 5.50          | 5.60             |
| 6.70          | 6.82             |
| 7.40          | 7.09             |

> The model showed consistent and acceptable prediction accuracy, indicating its potential for real-world usage where user-generated movie ratings are not yet available.

---

### 3. `GAIH_ML_Bootcamp_May2025_Tfidf-Cosine-MovieRecoSystemProject.ipynb`  
**Techniques Used**:  
- TF-IDF Vectorization & Cosine Similarity  
- Weighted Average Rating  
- Content-Based Filtering using Movie Overviews

**Goals**:  
- Recommend similar movies based on movie descriptions, keywords, taglines, and genre information.

**Results**:

**a. CombinedText Recommendation Model** (using `keywords`, `tagline`, `genres`):

Top 15 movies similar to *Iron Man*:

1. Iron Man 2  
2. Avengers: Age of Ultron  
3. Iron Man 3  
4. Guardians of the Galaxy  
5. Hostage  
6. Baahubali: The Beginning
7. Beyond the Black Rainbow  
8. Mystery Men
9. Cradle 2 the Grave  
10. Resident Evil: Extinction  
11. Unsullied 
12. Journey to the Center of the Earth  
13. Equilibrium  
14. Dune 
15. Street Fighter

---

**b. WeightedAverageRating Recommender** (based on `vote_count`, `vote_average`):

Top weighted movies similar to *Iron Man*:

1. Guardians of the Galaxy 
2. The Avengers  
3. Iron Man 3  
4. Guardians of the Galaxy  
5. Hostage  
6. Baahubali: The Beginning
7. Beyond the Black Rainbow  
8. Mystery Men
9. Cradle 2 the Grave  
10. Resident Evil: Extinction  
11. Unsullied 
12. Journey to the Center of the Earth  
13. Equilibrium  
14. Dune 
15. Street Fighter

`['Guardians of the Galaxy', 'The Avengers', 'Deadpool', 'Iron Man', 'Captain America: The Winter Soldier', 'X-Men: Days of Future Past', 'Avengers: Age of Ultron', 'Captain America: Civil War', 'X-Men: First Class', 'Iron Man 3', 'Ant-Man', 'Divergent', 'Spider-Man', ...]`

---

**c. Content-Based Recommender** (based on `overview` text only):

Top 15 Recommendations for *Iron Man*:

1. Iron Man 2  
2. Avengers: Age of Ultron  
3. Iron Man 3  
4. Ant-Man  
5. Transformers  
6. Spider-Man 2  
7. Star Wars: Clone Wars  
8. Guardians of the Galaxy  
9. The Avengers  
10. Deadpool  
11. Captain America: The Winter Soldier  
12. X-Men: Days of Future Past  
13. Captain America: Civil War  
14. X-Men: First Class  
15. Divergent

---

## 🧠 Algorithms Chosen & Why

*Used in:* `GAIH_ML_Bootcamp_May2025_MLMovieRecoSystemProject.ipynb`
- **KNN** was chosen for collaborative filtering due to its effectiveness in sparse rating matrices and ease of implementation.

*Used in:* `GAIH_ML_Bootcamp_May2025_MLMoviePredProject.ipynb`
- **Linear Regression** was used for rating prediction to establish a baseline model with interpretable outputs.
- **RandomForestRegressor**: Chosen for its robustness and ability to model nonlinear relationships. It reduces overfitting by averaging multiple decision trees and provides useful feature importance scores.
- **GradientBoostingRegressor**: Used for its sequential learning approach that improves predictions by focusing on previous errors. Works well on structured data with proper tuning.
- **XGBRegressor**: Preferred for its speed and regularization capabilities. Offers high performance on tabular data and prevents overfitting better than traditional Gradient Boosting.
- **RandomizedSearchCV**: Applied to efficiently search a large hyperparameter space by randomly sampling combinations, saving time during tuning.
- **GridSearchCV**: Used for exhaustive search on smaller, well-defined parameter grids to find the optimal model configuration.
All models were evaluated using cross-validation to ensure reliable and generalizable performance.

*Used in:* `GAIH_ML_Bootcamp_May2025_MLMoviePredProject.ipynb`
- **TF-IDF + Cosine Similarity** and **Content-Based Filtering** were selected to demonstrate real-world application of NLP techniques in recommendation systems.

Each model was tested and evaluated on suitable metrics like cosine similarity score, rating prediction accuracy, and qualitative result interpretation.

---

## 🌍 Real-World Impact

This system can be integrated into a streaming platform or movie database to:

- Help users to save time during movie selection.
- Help users discover new movies based on their viewing history.
- Provide movie suggestions in cold-start scenarios (e.g., for new users).
- Predict audience feedback or IMDb scores before official ratings are collected.

---

## 🔮 Future Improvements

- Add **hybrid recommender system** combining collaborative and content-based approaches.
- Improve rating prediction with **ensemble methods** (e.g., XGBoost, Random Forest).
- Apply **deep learning techniques** like AutoEncoders or Transformers for better recommendation quality.
- Deploy the application via **Streamlit** or a web interface for public usage.
- Expand dataset using web scraping or external movie APIs like OMDB or TMDB.

---

## 📊 Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- NLTK / TF-IDF  
- Jupyter Notebook  
- Kaggle Notebook Environment  

---

## 📬 Contact

If you have any questions, feel free to reach out via GitHub issues or open a pull request to contribute ideas for improvement.

---

> _This project was developed by CodingWithMK (Musab Kaya) as part of the Akbank & Global AI Hub Machine Learning Bootcamp - May 2025 Cohort._
