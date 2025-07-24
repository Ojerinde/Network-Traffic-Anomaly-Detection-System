if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser(
        description="Network Anomaly Detection Entry Point")
    argument_parser.add_argument("--mode", choices=["supervised", "unsupervised", "both"], required=True,
                                 help="Specify which pipeline to run: supervised, unsupervised, or both")
    parsed_arguments = argument_parser.parse_args()

    if parsed_arguments.mode == "supervised":
        from utils.supervised_pipeline import run_supervised_pipeline
        run_supervised_pipeline()

    elif parsed_arguments.mode == "unsupervised":
        from utils.unsupervised_pipeline import run_unsupervised_pipeline
        run_unsupervised_pipeline()

    elif parsed_arguments.mode == "both":
        from utils.supervised_pipeline import run_supervised_pipeline
        from utils.unsupervised_pipeline import run_unsupervised_pipeline
        from utils.visualizer import NetworkAnalyzerVisualizer

        supervised_results = run_supervised_pipeline(create_visuals=True)
        unsupervised_results = run_unsupervised_pipeline(create_visuals=True)

        comparison_visualizer = NetworkAnalyzerVisualizer()
        comparison_visualizer.create_model_comparison(
            supervised_results, unsupervised_results)

        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Supervised Model ({supervised_results['model_name']}):")
        print(
            f"  • Accuracy: {supervised_results['classification_report'].get('accuracy', 0.0):.3f}")
        print(f"  • Test samples: {supervised_results['test_size']:,}")
        print(f"\nUnsupervised Model ({unsupervised_results['model_name']}):")
        print(
            f"  • Anomaly detection rate: {unsupervised_results['anomaly_count']/unsupervised_results['test_size']:.1%}")
        print(
            f"  • Anomalies detected: {unsupervised_results['anomaly_count']:,}")
        print(f"  • Test samples: {unsupervised_results['test_size']:,}")
