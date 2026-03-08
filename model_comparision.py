import pandas as pd
import os
import model_training  # This will trigger the training in model_training.py


def run_comparison():
    # Retrieve the results dictionary from your training script
    # This assumes model_training.py has a 'results' dictionary
    results = model_training.results

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    # Create the DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Validation Accuracy': [results[m]['val_accuracy'] for m in results.keys()],
        'ROC AUC': [results[m]['roc_auc'] for m in results.keys()]
    })

    print("\n", comparison_df)

    if not comparison_df.empty:
        # Identify the best model based on Validation Accuracy
        best_model_name = comparison_df.loc[comparison_df['Validation Accuracy'].idxmax(), 'Model']

        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"Validation Accuracy: {results[best_model_name]['val_accuracy']:.4f}")
        print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")

        # Path of the best model generated during training
        best_model_filename = f"malaria_{best_model_name.lower()}.h5"
        production_filename = "malaria_model.h5"

        # Rename the best model to the default name expected by app.py
        if os.path.exists(best_model_filename):
            # Remove existing production model if it exists to avoid errors on some OS
            if os.path.exists(production_filename):
                os.remove(production_filename)

            os.rename(best_model_filename, production_filename)
            print(f"✅ Best model saved as '{production_filename}' for production.")
        else:
            print(f"⚠️ Warning: Could not find {best_model_filename} to rename.")


if __name__ == "__main__":
    run_comparison()