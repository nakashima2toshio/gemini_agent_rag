from fastembed import SparseTextEmbedding
import logging

logging.basicConfig(level=logging.INFO)

try:
    print("Attempting to load 'hotchpotch/japanese-splade-base-v1'...")
    model = SparseTextEmbedding(model_name="hotchpotch/japanese-splade-base-v1")
    print("Success!")
except Exception as e:
    print(f"Failed to load custom model: {e}")

try:
    print("Attempting to load 'prithivida/Splade_PP_en_v1' (default)...")
    model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    print("Success!")
except Exception as e:
    print(f"Failed to load default model: {e}")
