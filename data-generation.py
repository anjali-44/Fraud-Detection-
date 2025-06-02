import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)

# Number of samples
num_samples = 1000
user_ids = np.random.randint(1, 50, num_samples)  # Random user IDs

data = {
    'Transaction_ID': range(1, num_samples + 1),
    'amount': np.random.uniform(10, 1000, num_samples),  # Transaction amounts
    'user_id': user_ids,
    'device': np.random.choice(['mobile', 'desktop', 'tablet'], num_samples),  # Devices
    'label': np.random.choice([0, 1], num_samples, p=[0.9, 0.1])  # 10% fraud cases
}

df = pd.DataFrame(data)

# Add time_of_day feature
df['time_of_day'] = np.random.choice(['morning', 'afternoon', 'evening', 'night'], num_samples)

# Add transaction_frequency feature (number of transactions per user)
user_transaction_counts = {user_id: np.random.randint(1, 20) for user_id in set(user_ids)}
df['transaction_frequency'] = df['user_id'].map(user_transaction_counts)

# Save the dataset
df.to_csv('synthetic_data.csv', index=False)
print("Synthetic data saved to 'synthetic_data.csv'")
