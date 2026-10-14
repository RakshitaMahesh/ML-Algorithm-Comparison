import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
rows = []

ads = pd.read_csv(os.path.join(DATA, "Social_Network_Ads.csv"))
X, y = ads.drop(columns=["Purchased", "User ID"]), ads["Purchased"]
cat = X.select_dtypes(include=["object"]).columns.tolist()
num = X.select_dtypes(exclude=["object"]).columns.tolist()
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}
for name, model in models.items():
    pipe = Pipeline([("preprocessing", pre), ("model", model)])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    rows.append({"Dataset":"Social Network Ads","Task":"Classification/Boosting","Algorithm":name,
                 "Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred,zero_division=0),
                 "Recall":recall_score(yte,pred,zero_division=0),"F1-score":f1_score(yte,pred,zero_division=0)})

def regression_experiment(path, target, dataset_name, models):
    df = pd.read_csv(path)
    X = df.drop(columns=[target])
    y = df[target]
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = X.select_dtypes(exclude=["object"]).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    for name, model in models.items():
        pipe = Pipeline([("preprocessing", prep), ("model", model)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rows.append({"Dataset":dataset_name,"Task":"Regression","Algorithm":name,
                     "MAE":mean_absolute_error(yte,pred),"RMSE":mean_squared_error(yte,pred)**0.5,"R2":r2_score(yte,pred)})

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}
regression_experiment(os.path.join(DATA,"insurance_pre.csv"), "charges", "Insurance", reg_models)
regression_experiment(os.path.join(DATA,"Salary_Data.csv"), "Salary", "Salary Data", {"Linear Regression":LinearRegression()})

cluster_X = ads[["Age","EstimatedSalary"]]
scaled = StandardScaler().fit_transform(cluster_X)
km = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = km.fit_predict(scaled)
pd.DataFrame({"Age":ads["Age"],"EstimatedSalary":ads["EstimatedSalary"],"Cluster":labels}).to_csv(os.path.join(OUT,"customer_clusters.csv"),index=False)
rows.append({"Dataset":"Social Network Ads","Task":"Clustering","Algorithm":"K-Means (k=2)",
             "Silhouette score":silhouette_score(scaled,labels),"Inertia":km.inertia_})
result = pd.DataFrame(rows)
result.to_csv(os.path.join(OUT,"model_comparison.csv"),index=False)
subset = result[result["Task"]=="Classification/Boosting"].sort_values("Accuracy",ascending=False)
plt.figure(figsize=(9,5))
plt.bar(subset["Algorithm"],subset["Accuracy"])
plt.ylabel("Accuracy")
plt.title("Classification and Boosting Model Accuracy")
plt.ylim(0,1)
plt.xticks(rotation=25,ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"classification_accuracy.png"),dpi=160)
print(result.to_string(index=False))
print("\\nSaved results to:", OUT)
