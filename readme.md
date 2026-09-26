# 🛒 SmartCart — Customer Segmentation Using Clustering

## 📌 Project Overview

**SmartCart** is an unsupervised machine learning project that segments customers based on their **demographics, purchasing behavior, spending patterns, and engagement with different shopping channels**.

The project applies data preprocessing, feature engineering, exploratory data analysis, dimensionality reduction, and clustering techniques to identify distinct customer groups.

The main objective is to understand customer behavior and create meaningful segments that can support data-driven marketing and customer relationship strategies.

---

## 🎯 Objectives

The key objectives of this project are:

* Clean and preprocess customer data.
* Handle missing values and outliers.
* Engineer meaningful customer-level features.
* Transform categorical variables into numerical representations.
* Scale the dataset for machine learning.
* Reduce dimensionality using **PCA**.
* Determine an appropriate number of customer clusters.
* Apply **K-Means Clustering**.
* Apply **Agglomerative Clustering**.
* Analyze and characterize the resulting customer segments.

---

## 📊 Dataset

The project uses a customer dataset containing **2,240 customer records and 22 original features**.

The dataset includes information related to:

* Income
* Year of birth
* Education
* Marital status
* Children
* Customer joining date
* Recency
* Product spending
* Deal purchases
* Web purchases
* Catalog purchases
* Store purchases
* Web visits
* Campaign response
* Complaints

---

# 🔧 Data Preprocessing

## Missing Values

The `Income` column contained missing values.

Missing income values were replaced using the **median income**:

```python
df["Income"] = df["Income"].fillna(df["Income"].median())
```

---

## Feature Engineering

Several new features were created to better represent customer behavior.

### Age

Customer age was calculated from the year of birth:

```python
df["Age"] = 2026 - df["Year_Birth"]
```

### Customer Tenure

The customer joining date was converted into a datetime format, and customer tenure was calculated in days.

```python
Customer_Tenure_Days =
reference_date - Dt_Customer
```

### Total Spending

A new `Total_Spending` feature was created by combining spending across:

* Wines
* Fruits
* Meat products
* Fish products
* Sweet products
* Gold products

### Total Children

`Kidhome` and `Teenhome` were combined:

```python
Total_Children = Kidhome + Teenhome
```

### Education

Education categories were consolidated into:

* Undergraduate
* Graduate
* Postgraduate

### Living With

Marital status was transformed into a simpler feature:

* `Partner`
* `Alone`

---

# 🧹 Removing Unnecessary Features

The following original columns were removed after feature engineering:

* `ID`
* `Year_Birth`
* `Marital_Status`
* `Kidhome`
* `Teenhome`
* `Dt_Customer`

The individual product spending columns were also removed after creating `Total_Spending`.

The resulting dataset contained **15 features** before encoding.

---

# 🚨 Outlier Treatment

Potential outliers were explored using pair plots.

The project removed observations based on:

```python
Age < 90
Income < 600000
```

Dataset size:

| Stage                 | Records |
| --------------------- | ------: |
| Original dataset      |   2,240 |
| After outlier removal |   2,236 |

Only **4 observations** were removed during this step.

---

# 🔤 Categorical Encoding

The categorical features:

* `Education`
* `Living_With`

were converted into numerical variables using **One-Hot Encoding**.

After encoding, the dataset contained:

**2,236 observations × 18 features**

---

# 📏 Feature Scaling

Because clustering algorithms are sensitive to feature scale, the features were standardized using `StandardScaler`.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

# 📉 Dimensionality Reduction — PCA

**Principal Component Analysis (PCA)** was used to reduce the dimensionality of the standardized dataset.

Three principal components were retained:

```python
PCA(n_components=3)
```

The explained variance ratios were:

```text
PCA 1 → 23.16%
PCA 2 → 11.39%
PCA 3 → 10.41%
```

Together, the three components explain approximately **44.55% of the variance** in the processed dataset.

A 3D visualization was created using the three principal components.

---

# 🔢 Determining the Number of Clusters

Two approaches were used to investigate the appropriate number of clusters.

## 1. Elbow Method

The **Within-Cluster Sum of Squares (WCSS)** was calculated for values of K from 1 to 10.

The `KneeLocator` technique was then used to identify the elbow point.

The resulting optimal K was:

```text
K = 4
```

---

## 2. Silhouette Score

Silhouette scores were calculated for cluster counts ranging from:

```text
K = 2 → K = 10
```

The silhouette analysis was used as an additional measure of cluster structure and separation.

---

# 🤖 Clustering Algorithms

## K-Means Clustering

K-Means clustering was applied using four clusters:

```python
kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

labels_kmeans = kmeans.fit_predict(X_pca)
```

The resulting clusters were visualized in three-dimensional PCA space.

---

## Agglomerative Clustering

The project also applies **Agglomerative Hierarchical Clustering**.

The configuration used was:

```python
AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)
```

The resulting customer clusters were visualized using the PCA components.

---

# 📊 Cluster Characterization

After clustering, the customer data was grouped by cluster and the mean values of the features were calculated.

The analysis examined differences between clusters across:

* Income
* Recency
* Deal purchases
* Web purchases
* Catalog purchases
* Store purchases
* Web visits
* Campaign response
* Age
* Customer tenure
* Total spending
* Number of children
* Education
* Living arrangement

### Example observations from the cluster analysis

The resulting clusters show differences in customer spending and purchasing behavior.

For example:

* Some clusters have average income around **₹36K–₹40K** while others are around **₹70K+**.
* Total spending varies considerably between the clusters.
* Higher-spending clusters also show substantially higher web, catalog, and store purchasing activity.
* Campaign response rates differ between customer segments.
* The number of children and living arrangements also vary between clusters.

These differences provide the basis for interpreting the customer segments.

---

# 📈 Visualizations

The project includes visualizations for:

* Feature distributions
* Pair plots
* Correlation heatmap
* PCA 3D projection
* Elbow curve
* Silhouette score analysis
* K-Means clusters
* Agglomerative clusters
* Cluster size distribution
* Income vs. Total Spending by cluster

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* **Pandas** — Data manipulation
* **NumPy** — Numerical operations
* **Matplotlib** — Data visualization
* **Seaborn** — Statistical visualization
* **Scikit-learn** — Preprocessing, PCA and clustering
* **Kneed** — Detecting the elbow point

### Development Environment

* Jupyter Notebook
* Git
* GitHub

---

# 📁 Project Structure

```text
Smartcart_Project/
│
├── smartcart.ipynb
├── smartcart_customers.csv
├── README.md
└── .gitignore
```

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/samT053/Smartcart_Project.git
```

## 2. Navigate to the Project

```bash
cd Smartcart_Project
```

## 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed
```

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open the Notebook

Open:

```text
smartcart.ipynb
```

Make sure `smartcart_customers.csv` is in the expected project directory before running the notebook.

---

# 💡 Business Applications

Customer segmentation can help businesses understand different customer groups and design strategies around their behavior.

Potential applications include:

### 🎯 Targeted Marketing

Design different campaigns for different customer segments.

### 💰 Customer Value Analysis

Identify groups with significantly different spending levels.

### 🛍️ Personalized Promotions

Use purchasing behavior to develop segment-specific offers.

### 📱 Channel Strategy

Understand differences between web, catalog, and store purchasing behavior.

### 🤝 Customer Engagement

Identify customer groups with different campaign response and engagement patterns.

---

# 🔮 Future Improvements

Potential extensions of this project include:

* Compare clustering quality using additional evaluation metrics.
* Perform deeper profiling of each customer segment.
* Experiment with **DBSCAN**.
* Compare K-Means and Agglomerative Clustering more systematically.
* Build an interactive customer segmentation dashboard using **Power BI**.
* Develop segment-specific marketing recommendations.
* Deploy the segmentation model as an application or API.

---

# 👨‍💻 Author

**Sam**

Data Analytics & Machine Learning

### Skills Demonstrated

* Python
* Pandas
* NumPy
* Exploratory Data Analysis
* Feature Engineering
* Data Visualization
* Data Preprocessing
* PCA
* K-Means Clustering
* Agglomerative Clustering
* Unsupervised Machine Learning
* Git & GitHub
