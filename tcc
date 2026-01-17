# Instalação de pacotes
!pip install -q catboost spacy nltk vaderSentiment seaborn
!python -m spacy download en_core_web_sm
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Imports
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from nltk.corpus import stopwords
import spacy
from sklearn.preprocessing import LabelEncoder

# Config
SEED = 42
sns.set(style="whitegrid")
np.random.seed(SEED)

# Carregar dataset IMDB
df = pd.read_csv("/content/IMDB Dataset.csv")  # ajuste caminho se necessário
print(df.head())
# Pegar só 10.000 exemplos em vez de 50.000
df = df.sample(1000, random_state=SEED).reset_index(drop=True)

# Pré-processamento
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)       # remove tags HTML
    text = re.sub(r"http\S+|www.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)   # remove caracteres especiais
    doc = nlp(text)
    return " ".join([t.lemma_ for t in doc if t.lemma_ not in stop_words and not t.is_punct])

df["clean_review"] = df["review"].apply(clean_text)

# Labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["sentiment"])  # positivo=1, negativo=0

X = df["clean_review"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# Vetorização
tfidf = TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1,2))

# ------------------ SVM ------------------
svm_pipeline = Pipeline([
    ("tfidf", tfidf),
    ("svm", SVC(kernel="linear", probability=True, random_state=SEED))
])
svm_pipeline.fit(X_train, y_train)

y_pred_svm = svm_pipeline.predict(X_test)
y_proba_svm = svm_pipeline.predict_proba(X_test)[:,1]

print("\nSVM Report:\n", classification_report(y_test, y_pred_svm, target_names=le.classes_))

# ------------------ CatBoost ------------------
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, random_state=SEED)
cat.fit(X_train_tfidf, y_train)

y_pred_cat = cat.predict(X_test_tfidf).astype(int)
y_proba_cat = cat.predict_proba(X_test_tfidf)[:,1]

print("\nCatBoost Report:\n", classification_report(y_test, y_pred_cat, target_names=le.classes_))

# ------------------ Métricas comparativas ------------------
acc_svm = accuracy_score(y_test, y_pred_svm)
acc_cat = accuracy_score(y_test, y_pred_cat)

print(f"Acurácia SVM: {acc_svm:.4f} | Acurácia CatBoost: {acc_cat:.4f}")

# ------------------ Gráficos ------------------

# 1. Matrizes de confusão
fig, axes = plt.subplots(1,2, figsize=(12,5))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", ax=axes[0], cmap="Blues")
axes[0].set_title("Confusion Matrix - SVM")
axes[0].set_xlabel("Previsto"); axes[0].set_ylabel("Verdadeiro")

sns.heatmap(confusion_matrix(y_test, y_pred_cat), annot=True, fmt="d", ax=axes[1], cmap="Greens")
axes[1].set_title("Confusion Matrix - CatBoost")
axes[1].set_xlabel("Previsto"); axes[1].set_ylabel("Verdadeiro")
plt.show()

# 2. ROC
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
fpr_cat, tpr_cat, _ = roc_curve(y_test, y_proba_cat)
auc_svm = auc(fpr_svm, tpr_svm)
auc_cat = auc(fpr_cat, tpr_cat)

plt.figure(figsize=(7,6))
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={auc_svm:.3f})")
plt.plot(fpr_cat, tpr_cat, label=f"CatBoost (AUC={auc_cat:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("Curva ROC")
plt.legend()
plt.show()

# 3. Precision-Recall
prec_svm, rec_svm, _ = precision_recall_curve(y_test, y_proba_svm)
prec_cat, rec_cat, _ = precision_recall_curve(y_test, y_proba_cat)

plt.figure(figsize=(7,6))
plt.plot(rec_svm, prec_svm, label="SVM")
plt.plot(rec_cat, prec_cat, label="CatBoost")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# 4. Gráfico comparativo de acurácia e AUC
metrics_df = pd.DataFrame({
    "Modelo": ["SVM","CatBoost"],
    "Acurácia": [acc_svm, acc_cat],
    "AUC": [auc_svm, auc_cat]
})
metrics_df.plot(x="Modelo", kind="bar", figsize=(8,5))
plt.title("Comparação de Desempenho")
plt.ylabel("Score")
plt.show()

# 5. Curvas de aprendizado
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(7,5))
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, "o-", label="Treino")
    plt.plot(train_sizes, test_scores_mean, "o-", label="Validação")
    plt.title(title)
    plt.xlabel("Tamanho do treino")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)
    plt.show()

# Learning Curve para SVM
plot_learning_curve(svm_pipeline, "Learning Curve - SVM", X_train, y_train, cv=3, n_jobs=-1)

# Learning Curve para CatBoost
plot_learning_curve(cat, "Learning Curve - CatBoost", X_train_tfidf, y_train, cv=3, n_jobs=-1)
