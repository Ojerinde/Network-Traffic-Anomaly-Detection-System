import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os
from typing import Dict, Any, Optional


class NetworkAnalyzerVisualizer:
    def __init__(self, output_directory: str = "visuals"):
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.facecolor'] = 'white'

    def visualize_supervised_results(self, trained_model, test_features, test_labels, model_name: str = "Random Forest"):
        predictions = trained_model.predict(test_features)
        report_dictionary = classification_report(
            test_labels, predictions, output_dict=True)

        results = {
            'model_name': model_name,
            'predictions': predictions,
            'true_labels': test_labels,
            'classification_report': report_dictionary,
            'test_size': len(test_labels)
        }

        self._create_supervised_performance_plot(results)
        self._create_confusion_matrix_plot(results)
        self._create_class_distribution_plot(results)

        return results

    def visualize_unsupervised_results(self, trained_model, test_features, anomaly_predictions, model_name: str = "Isolation Forest"):
        binary_predictions = np.where(anomaly_predictions == -1, 0, 1)

        results = {
            'model_name': model_name,
            'predictions': anomaly_predictions,
            'binary_predictions': binary_predictions,
            'anomaly_count': np.sum(anomaly_predictions == -1),
            'normal_count': np.sum(anomaly_predictions == 1),
            'test_size': len(anomaly_predictions)
        }

        self._create_unsupervised_analysis_plot(results)
        return results

    def create_model_comparison(self, supervised_results: Dict, unsupervised_results: Dict):
        figure, ((axis1, axis2), (axis3, axis4)) = plt.subplots(
            2, 2, figsize=(16, 12))

        supervised_report = supervised_results['classification_report']
        class_names = ['attack', 'normal']

        if 'attack' in supervised_report and 'normal' in supervised_report:
            performance_metrics = ['Precision', 'Recall', 'F1-Score']
            attack_performance = [
                supervised_report['attack']['precision'],
                supervised_report['attack']['recall'],
                supervised_report['attack']['f1-score']
            ]
            normal_performance = [
                supervised_report['normal']['precision'],
                supervised_report['normal']['recall'],
                supervised_report['normal']['f1-score']
            ]
        else:
            class_keys = sorted(
                [key for key in supervised_report if key in ['attack', 'normal']])
            performance_metrics = ['Precision', 'Recall', 'F1-Score']
            if len(class_keys) >= 2:
                attack_performance = [
                    supervised_report[class_keys[0]]['precision'],
                    supervised_report[class_keys[0]]['recall'],
                    supervised_report[class_keys[0]]['f1-score']
                ]
                normal_performance = [
                    supervised_report[class_keys[1]]['precision'],
                    supervised_report[class_keys[1]]['recall'],
                    supervised_report[class_keys[1]]['f1-score']
                ]
            else:
                attack_performance = [0.0, 0.0, 0.0]
                normal_performance = [0.0, 0.0, 0.0]

        bar_positions = np.arange(len(performance_metrics))
        bar_width = 0.35

        axis1.bar(bar_positions - bar_width/2, attack_performance, bar_width,
                  label='Attack', color='#FF6B6B', alpha=0.8)
        axis1.bar(bar_positions + bar_width/2, normal_performance, bar_width,
                  label='Normal', color='#4ECDC4', alpha=0.8)

        axis1.set_xlabel('Metrics')
        axis1.set_ylabel('Score')
        axis1.set_title(
            f'{supervised_results["model_name"]} Performance by Class')
        axis1.set_xticks(bar_positions)
        axis1.set_xticklabels(performance_metrics)
        axis1.legend()
        axis1.grid(True, alpha=0.3)

        for i, value in enumerate(attack_performance):
            axis1.text(i - bar_width/2, value + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        for i, value in enumerate(normal_performance):
            axis1.text(i + bar_width/2, value + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        supervised_class_counts = pd.Series(
            supervised_results['true_labels']).value_counts()
        supervised_labels = supervised_class_counts.index.tolist()
        supervised_sizes = supervised_class_counts.values.tolist()
        supervised_colors = ['#FF6B6B', '#4ECDC4']

        wedges, texts, auto_texts = axis2.pie(supervised_sizes, labels=supervised_labels, colors=supervised_colors,
                                              autopct='%1.1f%%', shadow=True, startangle=90)
        axis2.set_title('Test Set Distribution\n(Supervised Learning)')

        for auto_text in auto_texts:
            auto_text.set_color('white')
            auto_text.set_fontweight('bold')

        unsupervised_labels = ['Normal', 'Anomaly']
        unsupervised_sizes = [
            unsupervised_results['normal_count'], unsupervised_results['anomaly_count']]
        unsupervised_colors = ['#45B7D1', '#FFA07A']

        wedges2, texts2, auto_texts2 = axis3.pie(unsupervised_sizes, labels=unsupervised_labels, colors=unsupervised_colors,
                                                 autopct='%1.1f%%', shadow=True, startangle=90)
        axis3.set_title(f'{unsupervised_results["model_name"]} Results')

        for auto_text in auto_texts2:
            auto_text.set_color('white')
            auto_text.set_fontweight('bold')

        supervised_accuracy = supervised_report.get('accuracy', 0.0)
        unsupervised_anomaly_rate = unsupervised_results['anomaly_count'] / \
            unsupervised_results['test_size']

        model_comparison_names = [f'Supervised\n({supervised_results["model_name"]})',
                                  f'Unsupervised\n({unsupervised_results["model_name"]})']
        model_comparison_scores = [
            supervised_accuracy, unsupervised_anomaly_rate]
        score_type_labels = ['Accuracy', 'Anomaly Rate']
        comparison_colors = ['#8E44AD', '#E67E22']

        comparison_bars = axis4.bar(
            model_comparison_names, model_comparison_scores, color=comparison_colors, alpha=0.8)
        axis4.set_ylabel('Score')
        axis4.set_title('Model Performance Comparison')
        axis4.set_ylim(0, 1)
        axis4.grid(True, alpha=0.3)

        for bar, score, label in zip(comparison_bars, model_comparison_scores, score_type_labels):
            bar_height = bar.get_height()
            axis4.text(bar.get_x() + bar.get_width()/2., bar_height + 0.01,
                       f'{label}\n{score:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/model_comparison_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(
            f"Model comparison saved to {self.output_directory}/model_comparison_analysis.png")

    def _create_supervised_performance_plot(self, results: Dict):
        classification_report_data = results['classification_report']

        figure, ((axis1, axis2), (axis3, axis4)) = plt.subplots(
            2, 2, figsize=(15, 12))

        class_keys = [key for key in classification_report_data.keys() if isinstance(
            classification_report_data[key], dict) and 'precision' in classification_report_data[key]]

        if len(class_keys) >= 2:
            performance_metrics = ['Precision', 'Recall', 'F1-Score']
            first_class_scores = [classification_report_data[class_keys[0]]['precision'],
                                  classification_report_data[class_keys[0]
                                                             ]['recall'],
                                  classification_report_data[class_keys[0]]['f1-score']]
            second_class_scores = [classification_report_data[class_keys[1]]['precision'],
                                   classification_report_data[class_keys[1]
                                                              ]['recall'],
                                   classification_report_data[class_keys[1]]['f1-score']]

            bar_positions = np.arange(len(performance_metrics))
            bar_width = 0.35

            axis1.bar(bar_positions - bar_width/2, first_class_scores, bar_width,
                      label=class_keys[0], alpha=0.8)
            axis1.bar(bar_positions + bar_width/2, second_class_scores, bar_width,
                      label=class_keys[1], alpha=0.8)
            axis1.set_xticks(bar_positions)
            axis1.set_xticklabels(performance_metrics)
            axis1.set_ylabel('Score')
            axis1.set_title('Performance Metrics by Class')
            axis1.legend()
            axis1.grid(True, alpha=0.3)

        overall_metric_names = ['Accuracy', 'Macro Avg F1', 'Weighted Avg F1']
        overall_metric_values = [
            classification_report_data.get('accuracy', 0.0),
            classification_report_data.get(
                'macro avg', {}).get('f1-score', 0.0),
            classification_report_data.get(
                'weighted avg', {}).get('f1-score', 0.0)
        ]

        overall_bars = axis2.bar(overall_metric_names, overall_metric_values, color=[
            '#3498DB', '#E74C3C', '#2ECC71'], alpha=0.8)
        axis2.set_ylabel('Score')
        axis2.set_title('Overall Model Performance')
        axis2.set_ylim(0, 1)
        axis2.grid(True, alpha=0.3)

        for bar, value in zip(overall_bars, overall_metric_values):
            bar_height = bar.get_height()
            axis2.text(bar.get_x() + bar.get_width()/2., bar_height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        if len(class_keys) >= 2:
            support_counts = [classification_report_data[key]
                              ['support'] for key in class_keys]
            colors = sns.color_palette("husl", len(class_keys))
            axis3.bar(class_keys, support_counts, color=colors, alpha=0.8)
            axis3.set_ylabel('Number of Samples')
            axis3.set_title('Test Set Class Distribution')
            axis3.grid(True, alpha=0.3)

        for i, value in enumerate(support_counts):
            axis3.text(i, value + max(support_counts) * 0.01, str(int(value)),
                       ha='center', va='bottom', fontweight='bold')

        axis4.text(0.05, 0.95, 'Model Performance Insights:', fontsize=14, fontweight='bold',
                   transform=axis4.transAxes, va='top')

        performance_insights = [
            f'• Total test samples: {results["test_size"]:,}',
            f'• Overall accuracy: {classification_report_data.get("accuracy", 0.0):.1%}',
            f'• Macro average F1: {classification_report_data.get("macro avg", {}).get("f1-score", 0.0):.3f}',
            f'• Model: {results["model_name"]}'
        ]

        for i, insight in enumerate(performance_insights):
            axis4.text(0.05, 0.8 - i*0.15, insight, fontsize=12,
                       transform=axis4.transAxes, va='top')

        axis4.set_xlim(0, 1)
        axis4.set_ylim(0, 1)
        axis4.axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/supervised_performance_detailed.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(
            f"Detailed supervised performance saved to {self.output_directory}/supervised_performance_detailed.png")

    def _create_confusion_matrix_plot(self, results: Dict):
        true_labels = results['true_labels']
        predicted_labels = results['predictions']

        confusion_matrix_data = confusion_matrix(true_labels, predicted_labels)

        plt.figure(figsize=(10, 8))

        unique_labels = sorted(
            np.unique(np.concatenate([true_labels, predicted_labels])))

        sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[
                        f'Predicted {label}' for label in unique_labels],
                    yticklabels=[f'Actual {label}' for label in unique_labels],
                    cbar_kws={'label': 'Count'})

        plt.title(
            f'Confusion Matrix - {results["model_name"]}\n', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        model_accuracy = results['classification_report'].get('accuracy', 0.0)
        plt.figtext(0.02, 0.02, f'Overall Accuracy: {model_accuracy:.3f}', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/confusion_matrix.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(
            f"Confusion matrix saved to {self.output_directory}/confusion_matrix.png")

    def _create_class_distribution_plot(self, results: Dict):
        true_labels = results['true_labels']
        predicted_labels = results['predictions']

        figure, (axis1, axis2) = plt.subplots(1, 2, figsize=(14, 6))

        true_label_counts = pd.Series(true_labels).value_counts()
        axis1.pie(true_label_counts.values, labels=true_label_counts.index, autopct='%1.1f%%',
                  shadow=True, startangle=90)
        axis1.set_title('Actual Class Distribution')

        predicted_label_counts = pd.Series(predicted_labels).value_counts()
        axis2.pie(predicted_label_counts.values, labels=predicted_label_counts.index, autopct='%1.1f%%',
                  shadow=True, startangle=90)
        axis2.set_title('Predicted Class Distribution')

        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/class_distribution_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(
            f"Class distribution analysis saved to {self.output_directory}/class_distribution_analysis.png")

    def _create_unsupervised_analysis_plot(self, results: Dict):
        figure, ((axis1, axis2), (axis3, axis4)) = plt.subplots(
            2, 2, figsize=(15, 12))

        detection_labels = ['Normal', 'Anomaly']
        detection_counts = [results['normal_count'], results['anomaly_count']]
        detection_colors = ['#45B7D1', '#FFA07A']
        pie_explosion = (0.05, 0.05)

        wedges, texts, auto_texts = axis1.pie(detection_counts, explode=pie_explosion, labels=detection_labels, colors=detection_colors,
                                              autopct='%1.1f%%', shadow=True, startangle=90)
        axis1.set_title(f'{results["model_name"]} Detection Results')

        for auto_text in auto_texts:
            auto_text.set_color('white')
            auto_text.set_fontweight('bold')

        total_samples = results['test_size']
        anomaly_detection_rate = results['anomaly_count'] / total_samples
        normal_detection_rate = results['normal_count'] / total_samples

        detection_statistics = ['Anomaly Rate', 'Normal Rate']
        detection_rates = [anomaly_detection_rate, normal_detection_rate]

        rate_bars = axis2.bar(detection_statistics, detection_rates, color=[
                              '#FFA07A', '#45B7D1'], alpha=0.8)
        axis2.set_ylabel('Rate')
        axis2.set_title('Detection Rate Analysis')
        axis2.set_ylim(0, 1)
        axis2.grid(True, alpha=0.3)

        for bar, rate in zip(rate_bars, detection_rates):
            bar_height = bar.get_height()
            axis2.text(bar.get_x() + bar.get_width()/2., bar_height + 0.01,
                       f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')

        sample_categories = ['Normal Samples', 'Anomalous Samples']
        sample_counts = [results['normal_count'], results['anomaly_count']]

        count_bars = axis3.bar(sample_categories, sample_counts, color=[
                               '#45B7D1', '#FFA07A'], alpha=0.8)
        axis3.set_ylabel('Number of Samples')
        axis3.set_title('Sample Count Distribution')
        axis3.grid(True, alpha=0.3)

        for bar, count in zip(count_bars, sample_counts):
            bar_height = bar.get_height()
            axis3.text(bar.get_x() + bar.get_width()/2., bar_height + max(sample_counts) * 0.01,
                       f'{count:,}', ha='center', va='bottom', fontweight='bold')

        axis4.text(0.05, 0.95, 'Unsupervised Learning Results:', fontsize=14, fontweight='bold',
                   transform=axis4.transAxes, va='top')

        analysis_insights = [
            f'• Total samples analyzed: {total_samples:,}',
            f'• Anomalies detected: {results["anomaly_count"]:,} ({anomaly_detection_rate:.1%})',
            f'• Normal patterns: {results["normal_count"]:,} ({normal_detection_rate:.1%})',
            f'• Model: {results["model_name"]}',
            f'• Detection approach: No labeled training data'
        ]

        for i, insight in enumerate(analysis_insights):
            axis4.text(0.05, 0.8 - i*0.12, insight, fontsize=11,
                       transform=axis4.transAxes, va='top')

        axis4.set_xlim(0, 1)
        axis4.set_ylim(0, 1)
        axis4.axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/unsupervised_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(
            f"Unsupervised analysis saved to {self.output_directory}/unsupervised_analysis.png")
