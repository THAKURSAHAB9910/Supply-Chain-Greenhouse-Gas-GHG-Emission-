import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score,
    roc_curve, auc
)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

file_path = r"C:\Users\Vishe\Downloads\SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv"
df = pd.read_csv(file_path)
df = df.dropna().reset_index(drop=True)

print("Columns in dataset:\n", df.columns.tolist())

numeric_cols = [
    "Supply Chain Emission Factors with Margins",
    "Supply Chain Emission Factors without Margins",
    "Margins of Supply Chain Emission Factors"
]

categorical_cols = [
    col for col in df.columns
    if col not in numeric_cols and df[col].dtype == "object"
]

print("\nCategorical Columns Used:", categorical_cols)
print("Numeric Columns Used:", numeric_cols)

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

X = df[categorical_cols]
y_reg = df["Supply Chain Emission Factors with Margins"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(10,6))
sns.heatmap(df[categorical_cols + ["Supply Chain Emission Factors with Margins"]].corr(),
            annot=True, cmap="Spectral")
plt.title("Feature Correlation Analysis")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)

rf_reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_train)
y_rf = rf_reg.predict(X_test)

regression_results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [
        mean_absolute_error(y_test, y_lr),
        mean_absolute_error(y_test, y_rf)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_lr)),
        np.sqrt(mean_squared_error(y_test, y_rf))
    ],
    "R2": [
        r2_score(y_test, y_lr),
        r2_score(y_test, y_rf)
    ]
})

print("\nREGRESSION RESULTS\n", regression_results)

sns.barplot(data=regression_results, x="Model", y="R2", hue="Model", legend=False)
plt.title("Regression Model Comparison (R²)")
plt.show()

threshold = y_reg.median()
df["Emission_Class"] = (y_reg > threshold).astype(int)

y_clf = df["Emission_Class"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_scaled, y_clf, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
}

clf_results = []
model_outputs = {}

for name, model in models.items():
    model.fit(Xc_train, yc_train)
    y_pred = model.predict(Xc_test)
    y_prob = model.predict_proba(Xc_test)[:,1]

    clf_results.append({
        "Model": name,
        "Accuracy": accuracy_score(yc_test, y_pred),
        "Precision": precision_score(yc_test, y_pred),
        "Recall": recall_score(yc_test, y_pred),
        "F1": f1_score(yc_test, y_pred),
        "MAP": average_precision_score(yc_test, y_prob)
    })

    model_outputs[name] = {"pred": y_pred, "prob": y_prob}

classification_results = pd.DataFrame(clf_results)
print("\nCLASSIFICATION RESULTS\n", classification_results)

sns.barplot(data=classification_results, x="Model", y="MAP", hue="Model", legend=False)
plt.title("Classification Performance (MAP)")
plt.show()

fig, axes = plt.subplots(1, len(model_outputs), figsize=(14,4))

for ax, (name, out) in zip(axes, model_outputs.items()):
    sns.heatmap(confusion_matrix(yc_test, out["pred"]),
                annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(name)

plt.tight_layout()
plt.show()

plt.figure()
for name, out in model_outputs.items():
    fpr, tpr, _ = roc_curve(yc_test, out["prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")

plt.plot([0,1],[0,1],"k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(7,5))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="Set1")
plt.title("Industry Clusters based on Emission Intensity")
plt.show()

print("\nFINAL BEST MODELS")
best_reg = regression_results.sort_values("R2", ascending=False).iloc[0]["Model"]
best_clf = classification_results.sort_values("MAP", ascending=False).iloc[0]["Model"]

print("Best Regression Model     :", best_reg)
print("Best Classification Model :", best_clf)
print("Best Clustering Model     : K-Means")
