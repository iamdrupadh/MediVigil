import sys
import traceback

print(f"Python Version: {sys.version}")

try:
    import mediapipe as mp
    print(f"Mediapipe imported from: {mp.__file__}")
except Exception as e:
    print("Failed to import mediapipe completely:")
    traceback.print_exc()

try:
    from mediapipe.python import _framework_bindings
    print("Successfully imported _framework_bindings!")
except Exception as e:
    print("\n--- ROOT CAUSE OF THE ERROR ---")
    traceback.print_exc()
