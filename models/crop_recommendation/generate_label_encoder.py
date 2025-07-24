import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load crop data
csv_path = '../../data/Crop_recommendation.csv'
df = pd.read_csv(csv_path)
labels = df['label'].unique()

# Fit the label encoder
le = LabelEncoder()
le.fit(labels)

# Save the encoder
with open('../label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print('label_encoder.pkl generated successfully!') 