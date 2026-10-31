
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_FILES = [
    ROOT / "models/holybro_isolation_forest.joblib",
    ROOT / "models/speedybee_isolation_forest.joblib",
    ROOT / "models/holybro_future_remaining_time_30s.joblib",
]

for path in MODEL_FILES:
    print("\n" + "=" * 65)
    print(path.name)
    print("=" * 65)

    if not path.exists():
        print("MODEL FILE NOT FOUND")
        continue

    model = joblib.load(path)

    print("Python type:", type(model))
    print("Module:", type(model).__module__)

    if hasattr(model, "named_steps"):
        print("\nPipeline steps:")
        for name, step in model.named_steps.items():
            print(f"  {name}: {type(step).__name__}")

    if hasattr(model, "feature_names_in_"):
        print("Feature names:", len(model.feature_names_in_))
        print("First 10:", list(model.feature_names_in_[:10]))

    if hasattr(model, "n_features_in_"):
        print("Input feature count:", model.n_features_in_)

    if isinstance(model, dict):
        print("\nDictionary keys:", list(model.keys()))

    print("\nAvailable prediction methods:")
    for method in ["predict", "predict_proba", "decision_function"]:
        print(f"  {method}: {hasattr(model, method)}")
