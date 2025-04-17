import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score
import numpy as np

# -----------------------
# 1. Données factices
# -----------------------

data = [
    ("Juan Carlos Fernández Gómez", "J. Carlos Fernández Gómez", 1),
    ("Ana María López Pérez", "Ana M. López Pérez", 1),
    ("Luis Alberto García Torres", "Luis A. García", 1),
    ("Juan Carlos Fernández Gómez", "Carlos Antonio Rodríguez", 0),
    ("Ana María López Pérez", "María Teresa Jiménez", 0),
    ("Luis Alberto García Torres", "Pedro Luis Martín", 0)
]

random.shuffle(data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)

# -----------------------
# 2. Embeddings FastText
# -----------------------

# Tu peux charger un modèle plus lourd si tu veux (es, es-en)
# Ici on utilise le modèle léger multilingue de FastText
from gensim.downloader import load
fasttext_model = load("fasttext-wiki-news-subwords-300")  # 300 dim

def embed_name(name):
    tokens = name.lower().split()
    vectors = [fasttext_model[w] for w in tokens if w in fasttext_model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(300)

# -----------------------
# 3. Dataset PyTorch
# -----------------------

class NamePairsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.embedded = [(embed_name(n1), embed_name(n2), label) for n1, n2, label in data]

    def __len__(self):
        return len(self.embedded)

    def __getitem__(self, idx):
        x1, x2, y = self.embedded[idx]
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)

dataset = NamePairsDataset(data)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------
# 4. Réseau Siamois
# -----------------------

class SiameseNet(nn.Module):
    def __init__(self, input_dim=300):
        super(SiameseNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        combined = torch.cat([h1, h2], dim=1)
        return self.classifier(combined)

model = SiameseNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# 5. Entraînement
# -----------------------

for epoch in range(20):
    model.train()
    losses = []
    for x1, x2, y in loader:
        optimizer.zero_grad()
        output = model(x1, x2)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch+1}, Loss: {np.mean(losses):.4f}")

# -----------------------
# 6. Évaluation
# -----------------------

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x1, x2, y in loader:
        output = model(x1, x2)
        preds = (output > 0.5).float()
        y_true.extend(y.numpy().flatten())
        y_pred.extend(preds.numpy().flatten())

print("\nAccuracy:", accuracy_score(y_true, y_pred))


# -----------------------
# 7. Prédiction sur données jamais vues
# -----------------------

test_pairs = [
    ("Juan Carlos Fernández Gómez", "Juan C. Fernández Gómez"),
    ("Luis Alberto García Torres", "Luis A. G. Torres"),
    ("Pedro José Ramírez Ruiz", "P. J. Ramírez Ruiz"),
    ("Carlos Mendoza", "Roberto Sánchez"),
    ("Ana María López Pérez", "Ana López P."),
    ("Jose Aldo", "Jose Aldo Fernandez G."),
    ("Luiz Fernandez Gomez","Luis Fernandes Gomes"),
    ("Pedro Almodovar Sanchez", "P. Almodovare Sanchais")
]

print("\n--- Nouvelles prédictions ---")
model.eval()
with torch.no_grad():
    for n1, n2 in test_pairs:
        x1 = torch.tensor(embed_name(n1), dtype=torch.float32).unsqueeze(0)
        x2 = torch.tensor(embed_name(n2), dtype=torch.float32).unsqueeze(0)
        prob = model(x1, x2).item()
        match = prob > 0.5
        print(f"{n1}  <>  {n2}  --> Probabilité de match : {prob:.2f} --> Match: {match}")
