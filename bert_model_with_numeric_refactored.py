import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from EarlyStopping import EarlyStopping

# --- Config ---
MAX_SAMPLES = 50000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load BERT ---
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(DEVICE)


# --- Load and preprocess dataset ---
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df = df.dropna(
        subset=['Keyword', 'Match Type', 'Campaign Name', 'Ad Group Name', 'Avg. CPC', 'Bid', 'Campaign Goal'])
    df = df[df['Avg. CPC'] > 0].head(MAX_SAMPLES)
    df['Bid'] = df['Bid'].str.extract(r'([\d\.]+)').astype(float)
    return df


# --- Feature Engineering ---
def add_features(df):
    df['keyword_length'] = df['Keyword'].apply(len)
    df['keyword_word_count'] = df['Keyword'].apply(lambda x: len(str(x).split()))
    df['campaign_name_length'] = df['Campaign Name'].apply(len)
    df['adgroup_name_length'] = df['Ad Group Name'].apply(len)
    df['minbid_ratio'] = pd.to_numeric(df['Min. Bid'], errors='coerce') / df['Bid']
    return df


# --- Normalize Numeric Features ---
def normalize_features(df, columns):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns].fillna(0))
    return torch.tensor(scaled, dtype=torch.float32)


# --- BERT Embeddings ---
def get_bert_embeddings(texts):
    encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        output = bert_model(**encoded)
    return output.last_hidden_state.mean(dim=1).cpu()


# --- One-Hot Encoding ---
def encode_one_hot(series):
    encoded = pd.get_dummies(series.dropna())
    return torch.tensor(encoded.values, dtype=torch.float32)


# --- Fully connected model ---
class FCModel(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.net(x)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + 1e-8)


# --- Train model ---
def train_model(model, X_train, y_train, X_test, y_test, epochs=60):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = RMSELoss()
    early_stopper = EarlyStopping(patience=10, min_delta=1e-4)

    train_losses, test_losses, r2_scores = [], [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train).squeeze()
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        model.eval()
        with torch.no_grad():
            test_preds = model(X_test).squeeze()
            test_loss = loss_fn(test_preds, y_test)
            r2 = r2_score(y_test.numpy(), test_preds.numpy())

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        r2_scores.append(r2)

        if early_stopper(test_loss.item()):
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: Train RMSE={loss.item():.4f}, Test RMSE={test_loss.item():.4f}, R²={r2:.4f}")

    return train_losses, test_losses, r2_scores,test_preds, model


def main():
    df = load_and_clean_data('Keyword_details_and_perf_1.csv')
    df = add_features(df)

    keywords = df['Keyword'].tolist()
    campaigns = df['Campaign Name'].tolist()
    adgroups = df['Ad Group Name'].tolist()

    numeric_cols = ['keyword_length', 'keyword_word_count', 'campaign_name_length', 'adgroup_name_length',
                    'minbid_ratio']
    numeric_tensor = normalize_features(df, numeric_cols)

    key_embed = get_bert_embeddings(keywords)
    camp_embed = get_bert_embeddings(campaigns)
    adgroup_embed = get_bert_embeddings(adgroups)
    match_type_ohe = encode_one_hot(df['Match Type'])
    campaign_goal_ohe = encode_one_hot(df['Campaign Goal'])

    X_tensor = torch.cat([key_embed, match_type_ohe, camp_embed, adgroup_embed, campaign_goal_ohe, numeric_tensor],
                         dim=1)

    y_vals = np.log1p(df['Bid'].astype(float) / (df['Avg. CPC'] + 1e-8))
    y_tensor = torch.tensor(y_vals.values, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    model = FCModel(input_dim=X_tensor.shape[1])
    train_losses, test_losses, r2_scores,test_preds, trained_model = train_model(model, X_train, y_train, X_test, y_test)

    # --- Plot ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(r2_scores, label='R² Score')
    plt.title('R² Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.scatter(y_test.numpy(), y_test.numpy() - test_preds.numpy())
    plt.xlabel("True Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.axhline(0, color='red')
    plt.show()

if __name__ == "__main__":
    main()
