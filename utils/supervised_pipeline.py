import joblib
from sklearn.ensemble import RandomForestClassifier
from utils.data_loader import load_data
from utils.preprocessing import preprocess
from utils.evaluate_models import evaluate_model
from utils.visualizer import NetworkAnalyzerVisualizer
import os


def run_supervised_pipeline(create_visuals=True):
    print("Running supervised training pipeline...")

    training_data, testing_data = load_data()
    training_features, training_labels, feature_scaler, _ = preprocess(
        training_data)
    testing_features, testing_labels, _, _ = preprocess(testing_data)

    print("Training Random Forest Classifier...")
    random_forest_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    random_forest_classifier.fit(training_features, training_labels)

    os.makedirs("models", exist_ok=True)
    joblib.dump(random_forest_classifier, "models/supervised_rf.pkl")
    joblib.dump(feature_scaler, "models/scaler.pkl")

    print("Evaluating model...")
    evaluate_model(random_forest_classifier, testing_features, testing_labels)

    supervised_results = None
    if create_visuals:
        print("\nGenerating visualizations...")
        visualization_generator = NetworkAnalyzerVisualizer()
        supervised_results = visualization_generator.visualize_supervised_results(
            trained_model=random_forest_classifier,
            test_features=testing_features,
            test_labels=testing_labels,
            model_name="Random Forest"
        )

        results_metadata = {
            'model_type': 'supervised',
            'model_name': 'Random Forest',
            'test_samples': len(testing_features),
            'features': training_features.shape[1],
            'training_samples': len(training_features)
        }
        supervised_results.update(results_metadata)

        print(
            f"✓ Supervised pipeline completed with {len(testing_features)} test samples")

    return supervised_results
