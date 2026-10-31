
import numpy as np
import pandas as pd

np.random.seed(42)

n = 20000

df = pd.DataFrame({
    "latency_ms": np.random.gamma(2, 20, n),
    "packet_loss": np.random.beta(1, 20, n) * 10,
    "cpu_utilization": np.random.uniform(10, 100, n),
    "bandwidth_utilization": np.random.uniform(20, 100, n),
    "active_users": np.random.randint(50, 15000, n),
    "signal_strength": np.random.uniform(-115, -55, n)
})

# Simulate fault risk from network conditions
risk_score = (
    (df["latency_ms"] > 70).astype(int)
    + (df["packet_loss"] > 1.5).astype(int)
    + (df["cpu_utilization"] > 85).astype(int)
    + (df["bandwidth_utilization"] > 90).astype(int)
    + (df["signal_strength"] < -100).astype(int)
)

# Convert the risk score into a fault probability
probability = 1 / (1 + np.exp(-(risk_score - 2.5) * 1.2))

df["network_fault"] = (
    np.random.random(n) < probability
).astype(int)

print(df.head())
print(df["network_fault"].value_counts(normalize=True))
df.to_csv("network_data.csv", index=False)
print("Data saved to network_data.csv successfully.")