import tensorflow as tf

# Check TensorFlow version (optional, for debugging)
print(f"TensorFlow version: {tf.__version__}")

# List all physical devices
print("All physical devices:")
for device in tf.config.list_physical_devices():
    print(device)

# List all logical devices
print("\nAll logical devices:")
for device in tf.config.list_logical_devices():
    print(device)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\nNumber of GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU Device: {gpu}")
else:
    print("No GPU devices detected.")

# Get detailed GPU information (if available)
if gpus:
    try:
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"Details for GPU {gpu}: {details}")
    except Exception as e:
        print(f"Error fetching GPU details: {e}")

# Display memory growth setting (useful for memory management)
if gpus:
    for gpu in gpus:
        try:
            memory_growth = tf.config.experimental.get_memory_growth(gpu)
            print(f"Memory growth for {gpu}: {memory_growth}")
        except Exception as e:
            print(f"Error fetching memory growth setting: {e}")
