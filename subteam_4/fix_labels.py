import numpy as np

# Raste define kar rahe hain
old_file = '/home/teaching/TEAM_22_DATASET/processed/processed/dynamics_labels_fixed.npy'
new_file = '/home/teaching/TEAM_22_DATASET/processed/processed/dynamics_labels_fixed_BINARY.npy'

print("⏳ Data load ho raha hai...")
data = np.load(old_file)

# Median calculate karna taaki data barabar 2 hisso mein bat jaye
median_val = np.median(data)
print(f"📊 Median Threshold: {median_val:.4f}")

# Jo median se bada hai wo 1.0 (Fast), jo chota hai wo 0.0 (Slow)
binary_data = (data > median_val).astype(np.float32)

ones_count = int(np.sum(binary_data))
zeros_count = len(binary_data) - ones_count

print(f"🎯 Class Balance: {ones_count} Fast (1.0) | {zeros_count} Slow (0.0)")

# Nayi file save kar do
np.save(new_file, binary_data)
print(f"✅ Nayi Binary file save ho gayi: {new_file}")