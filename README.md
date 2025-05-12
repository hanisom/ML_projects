✅ Objective
The model aims to predict the ratio between Bid and Avg. CPC (cost-per-click) for advertising keywords using both textual and numerical features. The target variable is:

![Screenshot 2025-05-13 at 0.34.43.png](..%2F..%2F..%2F..%2Fvar%2Ffolders%2F07%2Fnjf4_cl12kl42k8cskjgjnz00000gn%2FT%2FTemporaryItems%2FNSIRD_screencaptureui_447SzC%2FScreenshot%202025-05-13%20at%200.34.43.png)

🧩 Input Data
CSV file: Keyword_details_and_perf_1.csv

Features used:

Textual: Keyword, Campaign Name, Ad Group Name

Categorical: Match Type, Campaign Goal

Numerical: keyword_length, minbid_ratio, etc.

🔍 Preprocessing
Clean rows with missing data

Extract numeric values from the Bid field

Generate BERT embeddings for text columns using bert-base-multilingual-uncased

Apply one-hot encoding to categorical features

Standardize numeric features with StandardScaler

Concatenate all features into a single tensor for model input

🧠 Model Architecture
A simple fully connected neural network:

Input layer: size depends on concatenated embeddings

Two hidden layers: 512 and 256 neurons, ReLU + Dropout(0.4)

Output: 1 neuron (regression)

Custom loss: RMSE (root mean squared error)

⚙️ Training Details
Optimizer: Adam

Learning rate scheduler: ReduceLROnPlateau

Early stopping after 10 epochs with minimal improvement

Train/test split: 80/20 using train_test_split

Metric tracked: RMSE and R² score

📈 Evaluation
Logs RMSE loss and R² on both train and test sets

Visualizes loss and R² progression over epochs