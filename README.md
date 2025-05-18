https://gamma.app/docs/Predictive-Bid-Optimization-Model-d4jkkf46urfdq6b?mode=doc

✅ Objective
The model aims to predict the ratio between Bid and Avg. CPC (cost-per-click) for advertising keywords using both textual and numerical features. The target variable is:


🧩 Input Data
CSV file: Keyword_details_and_perf_1.csv

**Features used:**

Textual: Keyword, Campaign Name, Ad Group Name

Categorical: Match Type, Campaign Goal

Numerical: keyword_length,keyword_word_count, minbid_ratio


🔍 Preprocessing
Clean rows with missing data

Extract numeric values from the Bid field

Generate BERT embeddings for text columns using 'bert-base-multilingual-uncased'

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
