# Supply Chain Analytics: Comprehensive Project Report & Interview Prep

## 1. Project Evaluation: Is this a "Basic" Project?
**Absolutely not.** This is a highly advanced, production-grade project. 

Most "basic" or beginner ML projects simply load a dataset, use `train_test_split`, run `model.fit()`, and print the accuracy. They completely ignore real-world engineering challenges. 

Your project stands out as **Senior-Level** for the following reasons:
1. **Target Leakage Prevention:** You specifically engineered the system to remove variables like `product_price` and `order_item_quantity` from the sales prediction, and post-order variables from the delivery prediction. Beginners never do this; they build models that cheat.
2. **Dual-Model Architecture:** You built two distinct models (Regression for Sales, Classification for Risk) operating on the same underlying data structure.
3. **Idempotent Data Pipelines:** Your `preprocess.py` handles inference vs. training flawlessly, preventing crashes when target columns are missing in production. 
4. **Explainable AI (XAI):** You integrated **SHAP** to explain *why* a prediction was made. This is a highly sought-after skill in enterprise ML.
5. **Full-Stack MLOps:** You didn't just build a Jupyter Notebook. You built a **FastAPI** backend with lifespan management, a **Streamlit** UI frontend, and a **CI/CD pipeline** (GitHub Actions) with `flake8` linting.

**Conclusion:** If you present this correctly, interviewers will see you as someone who understands how to deploy ML into the real world, not just a student who knows how to call Scikit-Learn.

---

## 2. Resume Points

### For Machine Learning Engineer Roles
* **Architected a Dual-Model Supply Chain ML System:** Developed an XGBoost regressor for sales forecasting (R² ~ 0.69) and an XGBoost classifier for late delivery risk prediction (AUC-ROC 0.85).
* **Eliminated Target Leakage:** Engineered a robust preprocessing pipeline using Scikit-Learn `ColumnTransformer` that strictly separates training logic from inference logic, preventing data leakage from post-order variables.
* **Integrated Explainable AI (XAI):** Implemented SHAP (Shapley Additive exPlanations) to provide real-time, interpretable feature importance for both regression and classification predictions in the production UI.
* **Deployed MLOps & CI/CD Pipelines:** Packaged the ML system using FastAPI with asynchronous lifespan model loading, containerized via Docker, and enforced code quality using GitHub Actions and Flake8.

### For Software Development Engineer (SDE) Roles
* **Built a Scalable ML API Layer:** Developed a high-performance REST API using FastAPI and Pydantic, implementing asynchronous context managers to load ML artifacts at server startup rather than per-request.
* **Developed an Interactive Dashboard:** Created a frontend user interface using Streamlit and Plotly to consume backend predictions and visualize business intelligence metrics in real-time.
* **Implemented Robust CI/CD:** Configured GitHub Actions workflows for automated continuous integration, enforcing strict `flake8` linting rules and preventing build regressions.
* **Refactored Legacy Code:** Upgraded deprecated API calls, resolved sparse matrix edge cases in production, and modularized monolithic code into decoupled `config`, `preprocess`, and `train` modules.

### For Data Analyst / Data Scientist Roles
* **Conducted Supply Chain Data Analysis:** Analyzed thousands of logistical records to uncover insights around shipping modes, customer segments, and regional delivery delays using Plotly and pandas.
* **Designed Business Intelligence Dashboards:** Built an interactive Streamlit dashboard to monitor overall sales revenue, late delivery percentages, and predictive risk scoring.
* **Engineered Advanced Features:** Applied Frequency Encoding for high-cardinality categorical variables (like `Order Region` and `Category`) to improve model performance without expanding dimensionality.
* **Communicated Actionable Insights:** Translated complex ML probabilities into a clear "Traffic Light" risk gauge (High/Medium/Low) to empower operations teams to proactively manage supply chain bottlenecks.

---

## 3. Detailed Technical Report

### A. Problem Statement
Supply chain operations suffer from two main unpredictabilities: forecasting expected revenue from complex orders, and predicting which shipments will fail to meet their scheduled delivery dates. 

### B. System Architecture
The project is built on a modern, decoupled stack:
1. **Model Training (Offline):** Python scripts (`train.py`, `train_classifier.py`) that read raw CSV data, run the `preprocess.py` pipeline, and output serialized `.pkl` artifacts (the models, feature columns, and frequency maps).
2. **API Backend (Online):** A `FastAPI` server (`src/app.py`) that loads the `.pkl` files into system memory on startup (`app.state`) and exposes HTTP POST endpoints (`/predict`).
3. **User Interface (Online):** A `Streamlit` app (`app.py`) that acts as the control center for EDA and interactive predictions. 

### C. Advanced Preprocessing & Idempotency
One of the hardest parts of production ML is ensuring the data transformations applied during training match exactly what happens during live predictions (inference). 
* **Frequency Encoding:** High-cardinality strings (like `city` names) cannot be One-Hot Encoded because it creates too many columns. We used Frequency Encoding. During training, we calculate the frequencies and save them to `frequency_maps.pkl`. During inference, we load this file. If a new city appears, we safely handle it.
* **Idempotency:** The pipeline was designed so that if you run `preprocess(df)` twice, it doesn't crash. Furthermore, it intelligently knows whether it is in `is_training=True` mode (where it must calculate targets) or `is_training=False` mode (where target variables like `sales` do not exist yet because they are what we are trying to predict).

### D. Why XGBoost?
We used **Extreme Gradient Boosting (XGBoost)** because:
1. It handles tabular data (like CSVs) better than Deep Learning models.
2. It natively handles non-linear relationships (e.g., the relationship between a specific product category and delivery delay).
3. It provides excellent integration with TreeExplainer (SHAP) for fast explainability.

---

## 4. Interview Questions & Answers

### Easy Questions
**Q1: What is XGBoost and why did you choose it over Random Forest?**
> *Answer:* XGBoost is a gradient boosting framework based on decision trees. Unlike Random Forest, which builds deep trees independently in parallel, XGBoost builds shallow trees sequentially, where each new tree tries to correct the residual errors of the previous trees. It was chosen because it typically yields higher accuracy on tabular datasets and handles complex, non-linear feature interactions efficiently.

**Q2: What is FastAPI and what makes it fast?**
> *Answer:* FastAPI is a modern Python web framework for building APIs. It is fast because it is built on Starlette (for asynchronous web routing) and Pydantic (for fast data validation). It natively supports `async`/`await`, allowing it to handle concurrent requests efficiently without blocking.

**Q3: How did you evaluate the performance of your Classification model?**
> *Answer:* I used multiple metrics: Accuracy, AUC-ROC, Precision, and Recall. In a supply chain context, **Recall** is often the most critical metric because we want to catch as many true late deliveries as possible so the operations team can intervene, even if it means generating a few false positives (lower precision).

### Medium Questions
**Q4: Can you explain "Target Leakage" and how you prevented it in this project?**
> *Answer:* Target leakage occurs when a model is trained using features that will not be available at the time of prediction, essentially allowing the model to "cheat" during training. For example, predicting 'sales' using `product_price` and `order_quantity` is a perfect math formula, not ML. To prevent this, I built a strict preprocessing pipeline that explicitly drops post-order variables (like `actual_shipping_days`) from the Delivery Risk model, ensuring it only learns from data available at the exact moment the order is placed.

**Q5: What is SHAP, and how did you implement it?**
> *Answer:* SHAP (Shapley Additive exPlanations) is a game-theoretic approach to explain the output of machine learning models. It tells us exactly how much each feature contributed to a specific prediction. I implemented `shap.TreeExplainer` on the XGBoost model. Because SHAP requires numerical data, I had to ensure the data was passed through the Scikit-Learn `ColumnTransformer` first, and explicitly converted from a sparse matrix to a dense array (`.toarray()`) before calculating the SHAP values.

**Q6: How did you handle categorical variables with high cardinality?**
> *Answer:* Standard One-Hot Encoding creates a new column for every unique category, which would explode the dimensionality of features like "Order City" or "Category". Instead, I implemented **Frequency Encoding**. This replaces the categorical string with the frequency (percentage) of its occurrence in the training data. I saved these mappings to a pickle file (`frequency_maps.pkl`) so that during production inference, the exact same mappings are applied.

### Hard Questions
**Q7: Explain the concept of an "Idempotent" data pipeline and why it was necessary for your `preprocess.py` script.**
> *Answer:* Idempotency means that an operation can be applied multiple times without changing the result beyond the initial application. In my `preprocess.py`, this was critical for the log-transformations (e.g., converting `sales` to `sales_log`). If the script ran twice, it would try to take the log of the log, destroying the data. I implemented checks to ensure transformations only occur if the source column exists and the target column does not. Additionally, the pipeline dynamically adapts to `is_training=True` vs `is_training=False` to safely handle the absence of target labels during live inference.

**Q8: Describe how you optimized the FastAPI server for loading Machine Learning models in production.**
> *Answer:* Loading a `.pkl` model from disk is an expensive, blocking I/O operation. If you load the model inside the `/predict` route, the API will freeze for every single request. To solve this, I used FastAPI's `@asynccontextmanager` lifespan events. This loads the XGBoost model, the feature columns, and the SHAP explainer into memory (`app.state`) exactly once when the server boots up. The endpoints then reference these pre-loaded objects in RAM, allowing for millisecond response times and high concurrency.

**Q9: If you deploy this to production and a month later the model's accuracy drops significantly, how would you diagnose the problem?**
> *Answer:* This is known as Model Drift or Data Drift. I would first check for Data Drift by comparing the distribution of incoming API request features against the distribution of the original training data. Has a new `shipping_mode` been introduced? Have `product_prices` shifted due to inflation? Second, I would check Concept Drift: the underlying relationship between features and the target might have changed (e.g., a new logistics carrier changed how delays happen). To fix it, I would set up a monitoring system (like Evidently AI) to alert on distribution changes, and trigger a CI/CD workflow to retrain the model on the freshest data, evaluate it against the old model, and hot-swap the `.pkl` files if the new model performs better.
