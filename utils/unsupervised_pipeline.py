import joblib
from sklearn.ensemble import IsolationForest
from utils.data_loader import load_data
from utils.preprocessing import preprocess
from utils.evaluate_models import evaluate_model
from utils.visualizer import NetworkAnalyzerVisualizer


def run_unsupervised_pipeline(create_visuals=True):
    print("Running unsupervised training pipeline...")

    training_data, testing_data = load_data()
    training_features, _, feature_scaler, _ = preprocess(training_data)
    testing_features, _, _, _ = preprocess(testing_data)

    print("Training Isolation Forest...")
    isolation_forest_model = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    isolation_forest_model.fit(training_features)

    joblib.dump(isolation_forest_model, "models/unsupervised_if.pkl")

    anomaly_predictions = isolation_forest_model.predict(testing_features)

    print("Evaluating model...")
    evaluate_model(isolation_forest_model, testing_features,
                   anomaly_predictions, is_unsupervised=True)

    unsupervised_results = None
    if create_visuals:
        print("\nGenerating visualizations...")
        visualization_generator = NetworkAnalyzerVisualizer()
        unsupervised_results = visualization_generator.visualize_unsupervised_results(
            trained_model=isolation_forest_model,
            test_features=testing_features,
            anomaly_predictions=anomaly_predictions,
            model_name="Isolation Forest"
        )

        results_metadata = {
            'model_type': 'unsupervised',
            'test_samples': len(testing_features),
            'features': training_features.shape[1],
            'training_samples': len(training_features),
            'contamination_rate': 0.1
        }
        unsupervised_results.update(results_metadata)

        print(
            f"✓ Unsupervised pipeline completed with {len(testing_features)} test samples")

    return unsupervised_results
