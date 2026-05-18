import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pickle

# ── 1. Carregar dados ──────────────────────────────────────────────
df = pd.read_csv("data/train.csv")

# ── 2. Selecionar features relevantes ─────────────────────────────
features = ["HomePlanet", "CryoSleep", "Destination", "Age",
            "VIP", "RoomService", "FoodCourt", "ShoppingMall",
            "Spa", "VRDeck"]

df = df[features + ["Transported"]].dropna()

# ── 3. Converter colunas categóricas para número ───────────────────
le = LabelEncoder()
for col in ["HomePlanet", "Destination"]:
    df[col] = le.fit_transform(df[col])

df["CryoSleep"] = df["CryoSleep"].astype(int)
df["VIP"]       = df["VIP"].astype(int)
df["Transported"] = df["Transported"].astype(int)

# ── 4. Separar X e y ──────────────────────────────────────────────
X = df[features]
y = df["Transported"]

# ── 5. Treinar modelo ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── 6. Avaliar ────────────────────────────────────────────────────
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"AUC: {auc:.4f}")

# ── 7. Salvar modelo ──────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modelo salvo em model.pkl")