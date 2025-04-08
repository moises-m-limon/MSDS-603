import pandas as pd
import pickle
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load the dataset from Lab2 (I used the diabetes dataset)
diabetes = load_diabetes()
X = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
y = pd.DataFrame(data=diabetes.target, columns=['target'])
df_diabetes = pd.concat([X, y], axis=1)

# Split into train and test and load into data folder
train_data, test_data = train_test_split(df_diabetes, test_size=0.2, random_state=42)
train_data.to_csv('course files/Data/diabetes_train.csv', index=False)
test_data.to_csv('course files/Data/diabetes_test.csv', index=False)

# Get features
features = train_data.columns.tolist()
features.remove('target')

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features)
    ])

# Create and fit pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor)
])
pipeline.fit(train_data[features])

# Transform data
processed_train_features = pipeline.transform(train_data[features])
processed_test_features = pipeline.transform(test_data[features])

# Convert back to DataFrame
processed_train_df = pd.DataFrame(
    processed_train_features, 
    columns=[f'scaled_{col}' for col in features]
)
processed_test_df = pd.DataFrame(
    processed_test_features, 
    columns=[f'scaled_{col}' for col in features]
)
processed_train_df['target'] = train_data['target']
processed_test_df['target'] = test_data['target']

# Save preprocessed data and pipeline
processed_train_df.to_csv('course files/Data/processed_train_data.csv', index=False)
processed_test_df.to_csv('course files/Data/processed_test_data.csv', index=False)
with open('course files/Data/pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)