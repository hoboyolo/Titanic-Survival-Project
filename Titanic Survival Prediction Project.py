from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TKAgg")  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

# Make the CSV path reliable
df = pd.read_csv(Path(__file__).resolve().parent / "titanic.csv")

print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
df.head(10)
df.info()
df.describe()

# Make sure these columns are numeric
for col in ["Survived", "Age", "SibSp", "Parch", "Fare"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = pd.read_csv("titanic.csv")
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
df.head(10)
df.info()
df.describe()

# Visualize missing values
fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#e74c3c" if df[col].isnull().any() else "#2ecc71" for col in df.columns]
ax.barh(df.columns, df.isnull().sum(), color=colors)
ax.set_xlabel("Number of Missing Values")
ax.set_title("Missing Values per Column")
for i, v in enumerate(df.isnull().sum()):
    if v > 0:
        ax.text(v + 5, i, str(v), va="center", fontweight="bold")
plt.tight_layout()
plt.show()

# Survival distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
survived_counts = df["Survived"].value_counts()
axes[0].bar(["Did Not Survive (0)", "Survived (1)"], survived_counts.values,
            color=["#e74c3c", "#2ecc71"])
axes[0].set_title("Survival Count")
axes[0].set_ylabel("Count")
for i, v in enumerate(survived_counts.values):
    axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold")

# Pie chart
axes[1].pie(survived_counts.values, labels=["Did Not Survive", "Survived"],
            autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"],
            startangle=90, explode=(0.05, 0.05))
axes[1].set_title("Survival Percentage")

plt.suptitle("Target Variable Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\nSurvival rate: {df['Survived'].mean():.2%}")
print(f"Death rate: {1 - df['Survived'].mean():.2%}")


# Distribution of numerical features
numerical_cols = ["Age", "Fare", "SibSp", "Parch"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    axes[i].hist(df[col].dropna(), bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    axes[i].axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.2f}")
    axes[i].axvline(df[col].median(), color="green", linestyle="--", label=f"Median: {df[col].median():.2f}")
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].legend()

plt.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Box plots to check for outliers
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for i, col in enumerate(numerical_cols):
    sns.boxplot(y=df[col], ax=axes[i], color="#3498db")
    axes[i].set_title(f"{col} — Box Plot")

plt.suptitle("Outlier Detection via Box Plots", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Distribution of categorical features
categorical_cols = ["Pclass", "Sex", "Embarked"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for i, col in enumerate(categorical_cols):
    counts = df[col].value_counts()
    axes[i].bar(counts.index.astype(str), counts.values, color="#9b59b6", edgecolor="white")
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    for j, v in enumerate(counts.values):
        axes[i].text(j, v + 5, str(v), ha="center", fontweight="bold")

plt.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Survival rate by Sex
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# By Sex
survival_by_sex = df.groupby("Sex")["Survived"].mean()
axes[0].bar(survival_by_sex.index, survival_by_sex.values, color=["#3498db", "#e74c3c"])
axes[0].set_title("Survival Rate by Sex")
axes[0].set_ylabel("Survival Rate")
for i, v in enumerate(survival_by_sex.values):
    axes[0].text(i, v + 0.02, f"{v:.2%}", ha="center", fontweight="bold")

# By Pclass
survival_by_class = df.groupby("Pclass")["Survived"].mean()
axes[1].bar(survival_by_class.index.astype(str), survival_by_class.values,
            color=["#2ecc71", "#f39c12", "#e74c3c"])
axes[1].set_title("Survival Rate by Passenger Class")
axes[1].set_ylabel("Survival Rate")
for i, v in enumerate(survival_by_class.values):
    axes[1].text(i, v + 0.02, f"{v:.2%}", ha="center", fontweight="bold")

# By Embarked
survival_by_embarked = df.groupby("Embarked")["Survived"].mean()
axes[2].bar(survival_by_embarked.index, survival_by_embarked.values,
            color=["#1abc9c", "#9b59b6", "#e67e22"])
axes[2].set_title("Survival Rate by Embarkation Port")
axes[2].set_ylabel("Survival Rate")
for i, v in enumerate(survival_by_embarked.values):
    axes[2].text(i, v + 0.02, f"{v:.2%}", ha="center", fontweight="bold")

plt.suptitle("Survival Rate by Categorical Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()


# Survival by Sex and Pclass combined
fig, ax = plt.subplots(figsize=(8, 5))
survival_grouped = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack()
survival_grouped.plot(kind="bar", ax=ax, color=["#e74c3c", "#3498db"])
ax.set_title("Survival Rate by Class and Sex")
ax.set_ylabel("Survival Rate")
ax.set_xlabel("Passenger Class")
ax.set_xticklabels(["1st Class", "2nd Class", "3rd Class"], rotation=0)
ax.legend(title="Sex")
plt.tight_layout()
plt.show()

# Age distribution by survival
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

age_survived = df.dropna(subset=["Age", "Survived"])

axes[0].hist(
    [age_survived.loc[age_survived["Survived"] == 0, "Age"],
     age_survived.loc[age_survived["Survived"] == 1, "Age"]],
    bins=30,
    alpha=0.6,
    color=["#e74c3c", "#2ecc71"],
    label=["Did Not Survive", "Survived"],
    edgecolor="white"
)
axes[0].set_title("Age Distribution by Survival")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")
axes[0].legend()

sns.kdeplot(
    data=age_survived,
    x="Age",
    hue="Survived",
    ax=axes[1],
    palette=["#e74c3c", "#2ecc71"],
    fill=True
)
axes[1].set_title("Age Density by Survival")
axes[1].set_xlabel("Age")
axes[1].legend(title="Survived")

plt.tight_layout()
plt.show()

# Create FamilySize feature
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Family size distribution
family_counts = df["FamilySize"].value_counts().sort_index()
axes[0].bar(family_counts.index, family_counts.values, color="#3498db", edgecolor="white")
axes[0].set_title("Family Size Distribution")
axes[0].set_xlabel("Family Size")
axes[0].set_ylabel("Count")

# Survival rate by family size
survival_by_family = df.groupby("FamilySize")["Survived"].mean()
axes[1].bar(survival_by_family.index, survival_by_family.values, color="#2ecc71", edgecolor="white")
axes[1].set_title("Survival Rate by Family Size")
axes[1].set_xlabel("Family Size")
axes[1].set_ylabel("Survival Rate")
axes[1].axhline(y=df["Survived"].mean(), color="red", linestyle="--", label="Overall Rate")
axes[1].legend()

plt.tight_layout()
plt.show()

# Correlation heatmap of numerical features
numeric_df = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]].copy()
numeric_df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=1, ax=ax)
ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Correlation with target
target_corr = corr["Survived"].drop("Survived").sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in target_corr.values]
ax.barh(target_corr.index, target_corr.values, color=colors)
ax.set_title("Correlation with Survival")
ax.set_xlabel("Correlation Coefficient")
ax.axvline(x=0, color="black", linewidth=0.5)
plt.tight_layout()
plt.show()