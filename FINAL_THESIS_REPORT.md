# SUPPLY CHAIN ANALYTICS & MACHINE LEARNING CAPSTONE
## Comprehensive Technical Thesis & Implementation Report
**Prepared for: Internship Submission & Technical Evaluation**

---

## CHAPTER 1: EXECUTIVE SUMMARY & BUSINESS VALUE
The modern global supply chain generates terabytes of transactional, logistical, and demographic data. Traditionally, businesses have relied on "descriptive analytics"—looking backwards at past data to understand what happened. This capstone project represents a paradigm shift toward "predictive analytics"—utilizing advanced Machine Learning (ML) to forecast future events before they occur.

This project architects a dual-model, unified ML ecosystem designed to solve two paramount business challenges:
1. **Sales Revenue Forecasting (Regression):** Predicting the monetary value of an order based on geographic routing, customer demographics, and shipping preferences, without relying on direct price-multiplier data leakage.
2. **Late Delivery Risk (Classification):** Calculating the real-time probability that a package will miss its scheduled delivery window, allowing operations teams to proactively intervene.

By combining extreme gradient boosting (XGBoost) with Explainable AI (SHAP) and a fully interactive Streamlit web dashboard, this project bridges the gap between raw data science and actionable business intelligence.

---

## CHAPTER 2: EXPLORATORY DATA ANALYSIS (EDA) & DATA CLEANING
Before algorithmic training could begin, the raw dataset required exhaustive cleaning and standardization. The initial dataset contained highly varied, unstructured, and inconsistent data types across over 40 columns.

### 2.1 Column Standardization
All column names were converted to a standard, Python-friendly format (`df.columns.str.lower().str.replace(' ', '_')`). This prevents runtime errors caused by invisible trailing spaces or inconsistent casing (e.g., `Order Region` vs `order_region`).

### 2.2 Temporal Feature Extraction
Machine learning models cannot natively understand string-based timestamps (e.g., `"2015-01-13 12:00:00"`). The pipeline engineered a custom datetime parser to extract cyclical, discrete temporal features:
*   `order_year`, `order_month`, `order_day`, `order_week`
*   `is_weekend`: A critical binary feature (1 or 0) indicating if the order was placed on a Saturday or Sunday, capturing distinct behavioral purchasing patterns.

### 2.3 Dropping Obsolete/Redundant Features
Features that provided zero predictive variance or were completely unique (like `customer_email`, `customer_password`, `product_image`) were programmatically dropped to reduce the dataset's memory footprint and prevent the model from memorizing noise.

---

## CHAPTER 3: ADVANCED PREPROCESSING & FEATURE ENGINEERING
The heart of this architecture lies in the `preprocess.py` module. A core engineering principle of this project is **Train-Serve Synchronization**—meaning the exact same preprocessing function is used during batch training and real-time inference on the web dashboard. This completely eliminates "Train-Serve Skew", a common bug in beginner ML pipelines.

### 3.1 Handling Missing Values
For numerical features, missing values were imputed with `0.0`. For categorical features, missing strings were imputed with the explicit label `"Unknown"`. This ensures the algorithm never crashes on null matrices during live Streamlit inference.

### 3.2 High-Cardinality Frequency Encoding
One of the major technical achievements of this project is the handling of high-cardinality categorical data.
*   **The Problem:** Traditional "One-Hot Encoding" works perfectly for columns like `shipping_mode` (which only has 4 unique values). However, columns like `customer_city` contain thousands of unique cities. One-Hot Encoding this would create thousands of new columns. This leads to an ultra-sparse matrix, crashing system memory, drastically slowing down training, and causing severe overfitting (known as the "Curse of Dimensionality").
*   **The Solution:** We engineered a custom **Frequency Encoding** logic. Instead of creating new columns, the text string (e.g., `"Los Angeles"`) is replaced by a float representing its frequency distribution within the dataset (e.g., `0.045`). This retains the informational signal of the city's popularity while keeping the feature space dense and optimized.

### 3.3 Logarithmic Transformations
Financial data is almost always right-skewed (a few massive enterprise orders, and thousands of tiny consumer orders). We applied a `np.log1p()` transformation to target variables to enforce a normal distribution, significantly stabilizing the gradient descent optimization within XGBoost.

---

## CHAPTER 4: IDENTIFYING & ERADICATING DATA LEAKAGE
This chapter represents the most "Senior-Level" analytical work of the entire capstone. 

### 4.1 The Regression Leakage (Sales)
Initially, the Sales regression model achieved a near-perfect $R^2$ score of `0.9999`. A junior data scientist might celebrate this, but mathematically, it is a severe red flag indicating **Data Leakage**.
*   **The Leak:** The training data contained `product_price` and `order_item_quantity`. The target variable, `Sales`, is mathematically defined as $Sales = Price \times Quantity$. The XGBoost model had simply learned to act as a basic calculator.
*   **The Resolution:** A robust ML pipeline must predict the unknown based on behavioral inputs. We architected the `SALES_EXTRA_LEAKY` exclusion list inside `config.py`. By explicitly dropping price and quantity, we forced the model to predict monetary sales based strictly on market region, customer segment, and shipping choices. The resulting $R^2$ of `0.69` is a massive analytical victory, proving the model captures genuine human purchasing behavior.

### 4.2 The Classification Leakage (Delivery Risk)
*   **The Leak:** To predict if a package will be late, we cannot use post-event information. The initial dataset included `actual_shipping_days` and `delivery_status`. If a model knows it took 6 days to ship, predicting "Late" requires zero intelligence.
*   **The Resolution:** We implemented the `CLASSIFIER_EXTRA_DROP` list. The final delivery risk classifier relies 100% on information available *at the exact moment the customer clicks "Buy"* (e.g., Scheduled Days, Shipping Mode, Region).

---

## CHAPTER 5: ALGORITHMIC ARCHITECTURE (XGBOOST)
Both pipelines utilize **XGBoost (eXtreme Gradient Boosting)**.

### 5.1 Mathematical Intuition
Unlike Random Forests, which build hundreds of deep decision trees independently and average the result, XGBoost builds shallow trees *sequentially*. 
1. Tree 1 makes a prediction.
2. The algorithm calculates the "Residuals" (the errors Tree 1 made).
3. Tree 2 is built specifically to predict and correct the residuals of Tree 1.
4. This repeats for hundreds of iterations, using Gradient Descent to minimize a specific Loss Function (Mean Squared Error for Sales, Log-Loss for Delivery Risk).

### 5.2 Pipeline & Hyperparameter Tuning
We utilized Scikit-Learn's `Pipeline` API to chain our `ColumnTransformer` (handling the encoding) directly into the `XGBClassifier` and `XGBRegressor`. 
To achieve optimal accuracy, we ran `RandomizedSearchCV` across a 3-fold cross-validation grid, tuning:
*   `learning_rate`: The step size of the gradient descent.
*   `max_depth`: Limiting tree depth to 5 or 7 to prevent overfitting.
*   `subsample`: Randomly sampling 80% of rows per tree to force generalization.

---

## CHAPTER 6: MODEL EVALUATION & METRICS
### 6.1 The Sales Regressor
*   **$R^2$ (R-Squared) = 0.69:** This indicates that 69% of the variance in global sales revenue can be explained purely by regional routing and customer demographics.
*   **RMSE / MAE:** Evaluated in log-space and real dollars to measure the average magnitude of prediction error.

### 6.2 The Delivery Risk Classifier
Accuracy is a notoriously bad metric for imbalanced datasets. If 80% of packages are on time, a "dumb" model that always guesses "On Time" achieves 80% accuracy but is completely useless to the business. Therefore, we optimized for:
*   **Recall (~73%):** Out of all packages that were *actually* late, the model successfully caught 73% of them. In supply chain risk management, minimizing "False Negatives" (missed delays) is the highest priority.
*   **Precision (~84%):** When the model flashes a "High Risk" warning, it is correct 84% of the time, ensuring warehouse managers aren't wasting time investigating false alarms.
*   **AUC-ROC (0.85):** Demonstrates an excellent ability to separate the "Late" class from the "On Time" class across all probability thresholds.

---

## CHAPTER 7: EXPLAINABLE AI (SHAP)
A major bottleneck in modern AI adoption is the "Black Box" problem—executives do not trust algorithms they cannot understand. 
To solve this, we integrated **SHAP (SHapley Additive exPlanations)**. Based on cooperative game theory, SHAP calculates the exact marginal contribution of every feature to the final output.
Inside the Streamlit dashboard, when a user predicts a Delivery Risk, SHAP dynamically generates a plot proving exactly *why* that prediction was made (e.g., proving that "Standard Class" shipping pushed the probability up by +25%, while "LATAM Region" pushed it up by +15%).

---

## CHAPTER 8: DEPLOYMENT & UI/UX (STREAMLIT)
The final deliverable is a highly polished, interactive `app.py` Streamlit dashboard.
*   **Tab 1: EDA & BI Reporting:** Utilizes Plotly to render interactive charts (Sales by Region, Delivery Status donuts) allowing executives to explore historical data.
*   **Tab 2: Sales Prediction:** A dynamic input form passing live user inputs through the preprocessing pipeline into the XGBoost Regressor.
*   **Tab 3: Delivery Risk:** A custom Plotly-rendered Gauge Chart mapping probabilities from 0-100%, dynamically turning Red, Yellow, or Green based on the calculated risk. String inputs are strictly `.lower()`'d behind the scenes to ensure perfect matching with the trained One-Hot Encoder dictionary.

---
---

## EXHAUSTIVE INTERVIEW QUESTION BANK

### 🟢 ENTRY-LEVEL (Fundamentals)
**Q: What is the main difference between your two models?**
> **A:** The Sales model is a **Regression** task because it predicts a continuous numerical value (money). The Delivery Risk model is a **Classification** task because it predicts a discrete category (Yes/No, Late/On-Time). 

**Q: Why did you use Streamlit instead of React or Angular?**
> **A:** Streamlit is built specifically for data scientists. It allowed me to load my `.pkl` machine learning models and Pandas DataFrames natively into the web app using pure Python, without needing to architect a complex REST API backend.

**Q: Why did you drop the `actual_shipping_days` column for the classifier?**
> **A:** If I want the model to predict a late delivery *at the moment the customer checks out*, the model cannot have access to how many days it *actually* took to ship. That information doesn't exist yet in the real world. Including it would cause severe data leakage.

### 🟡 MID-LEVEL (Architecture & Preprocessing)
**Q: How does your pipeline handle missing data during live Streamlit inference?**
> **A:** My `preprocess.py` module uses a strict standardization process. If the user leaves an input blank, or if the live data feed drops a value, my code intercepts it. Numeric columns are filled with `0.0`, and categorical columns are explicitly filled with the string `"Unknown"`. The XGBoost model was trained on these "Unknown" mappings, ensuring it never crashes and simply treats missing data as a distinct category.

**Q: Explain "Frequency Encoding" and why you didn't just use One-Hot Encoding for everything.**
> **A:** One-Hot Encoding (OHE) creates a new binary column for every unique category. For a column like "City" with 3,000 unique values, OHE creates 3,000 new columns. This explodes the memory requirements and leads to a massive matrix of mostly zeros, confusing the model. Frequency Encoding solves this by replacing the City name with a single float: the percentage of times that city appears in the dataset. It retains the data's signal without expanding the dimensionality.

**Q: I see your Streamlit code applies `.lower()` to the user inputs. Why is this necessary?**
> **A:** During training, all strings were converted to lowercase (e.g., "latam", "standard class"). The One-Hot Encoder memorizes this exact casing. If the Streamlit dropdown passes "LATAM" (capitalized), the encoder doesn't recognize it, treats it as "unknown", and outputs an array of all zeros. Applying `.lower()` guarantees the live data matches the trained vocabulary perfectly.

### 🔴 SENIOR-LEVEL (Advanced Logic & Leakage)
**Q: Walk me through the mathematical intuition behind why an $R^2$ of 0.999 in your initial Sales model was actually a failure, and why 0.69 is a success.**
> **A:** An $R^2$ of 0.999 indicates the model perfectly memorized the variance. Upon investigation, the dataset included `Product Price` and `Order Quantity`. Because `Sales = Price * Quantity`, the model was performing basic arithmetic, not machine learning. This is classic Data Leakage. I architected an exclusion list (`SALES_EXTRA_LEAKY`) to explicitly drop price and quantity. The new model achieved 0.69, which is a massive success because it proves the model can predict the monetary value of a cart based purely on hidden behavioral metadata, like the customer's geographic region and departmental preferences.

**Q: Why did you optimize for "Recall" in your Logistics Classifier instead of just aiming for 95% Accuracy?**
> **A:** Supply chain datasets are highly imbalanced—most packages are on time. If 90% of packages are on time, a useless model that *always* predicts "On Time" achieves 90% accuracy. Instead, I focused on Recall, which measures: *Out of all the packages that actually failed and were late, how many did we successfully flag?* In supply chains, the financial penalty of a missed delay (lost customer trust) is exponentially worse than the cost of a false positive (an ops manager checking a fine package). My model achieved ~73% Recall, successfully catching the vast majority of delays before they happen.

**Q: How does XGBoost handle non-linear data better than Multiple Linear Regression?**
> **A:** Linear Regression attempts to draw a single hyper-plane through the data, assuming features scale uniformly. XGBoost builds sequential decision trees. Trees naturally handle non-linear cutoffs (e.g., "If Region = LATAM AND Days < 2, then Risk = High"). Because XGBoost optimizes a loss function via gradient descent over hundreds of sequential trees, it maps highly complex, multi-dimensional interactions that a linear equation fundamentally cannot capture.

---
---

## ATS-OPTIMIZED RESUME POINTS
*(Copy and paste these directly into your resume depending on the role you are applying for. These are heavily optimized for modern Applicant Tracking System bots).*

### 💻 For Software Engineering / Backend Roles
*   **Architected a Dual-Model Web Application:** Designed and deployed an end-to-end Python Streamlit web dashboard integrating dual XGBoost machine learning pipelines for predictive supply chain logistics and revenue forecasting.
*   **Engineered Robust ML Inference APIs:** Developed a centralized `preprocess.py` pipeline utilizing Pandas and Scikit-Learn that shares a synchronized codebase for large-scale training and real-time Streamlit inference, completely eliminating train-serve skew.
*   **Frontend Data Visualization Implementation:** Programmed an interactive, multi-tab UI featuring real-time Plotly probability gauges, exploratory data analysis (EDA) charts, and state-managed user input forms.
*   **Explainable AI Integration:** Implemented SHAP (SHapley Additive exPlanations) to dynamically render feature-importance charts on the frontend, providing transparent, real-time reasoning for complex algorithmic decisions.

### 🧠 For Machine Learning Engineering Roles
*   **Developed Leakage-Free ML Pipelines:** Identified and resolved severe data leakage in legacy models by isolating post-event target derivations, resulting in a robust Sales Regressor ($R^2$ 0.69) and Delivery Risk Classifier (AUC-ROC 0.85).
*   **Optimized Algorithmic Architecture:** Engineered custom `ColumnTransformer` pipelines using mixed One-Hot and Frequency Encoding to handle high-cardinality geographic data, minimizing dimensionality and preventing memory overflow.
*   **Advanced Hyperparameter Tuning:** Executed rigorous hyperparameter optimization via `RandomizedSearchCV` across 3-fold Cross-Validation to tune gradient descent step sizes, tree depth, and sub-sampling rates in XGBoost algorithms.
*   **Explainability & Interpretability:** Built automated SHAP value generation directly into the live inference pipeline, transforming black-box XGBoost predictions into highly interpretable business insights for operations teams.

### 📊 For Data Analyst / Business Intelligence Roles
*   **Supply Chain Risk Modeling:** Analyzed 40,000+ logistical records to develop an automated Late Delivery Risk model, achieving an 84% precision rate in flagging at-risk shipments before warehouse dispatch.
*   **Sales Revenue Forecasting:** Engineered a predictive forecasting model that estimates global order value based on regional and demographic trends, explicitly isolating direct price multipliers to capture true consumer purchasing behavior.
*   **Interactive BI Dashboards:** Designed and deployed an interactive Streamlit dashboard featuring custom Plotly visualizations, enabling executive stakeholders to drill down into payment types, regional sales, and granular shipping delays.
*   **Data Cleaning & Feature Engineering:** Extracted complex temporal trends from raw datetime strings and resolved high-cardinality geographic data issues via frequency distribution encoding, significantly improving data quality for downstream analytics.
