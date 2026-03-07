import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# 1. Import architectures from your model_building file
from model_building import create_custom_cnn, create_transfer_model


# CONFIGURATION
class Config:
    IMG_SIZE = (128, 128, 3)
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DATASET_PATH = r"D:\Amrita School of AI\SkillWallet\Maleria_Project\subset_dataset"


config = Config()

# 2. Data Generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    config.DATASET_PATH,
    target_size=(128, 128),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    config.DATASET_PATH,
    target_size=(128, 128),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


def plot_history(history, model_name):
    """Generates and saves training/validation plots for a specific model."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{model_name}_training_plots.png")
    print(f"📈 Saved plots for {model_name} to {model_name}_training_plots.png")
    plt.close()


def train_model(model, name, train_data, val_data, epochs, lr):
    print(f"\n🚀 Training Started for: {name}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='roc_auc')]
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=1
    )

    # Generate plots immediately after training
    plot_history(history, name)

    results = {
        'val_accuracy': history.history['val_accuracy'][-1],
        'roc_auc': history.history['val_roc_auc'][-1]
    }

    return history, model, results


# 3. Train multiple models
models_to_train = {
    'CustomCNN': create_custom_cnn(config.IMG_SIZE),
    'MobileNetV2': create_transfer_model('MobileNetV2', config.IMG_SIZE),
    'EfficientNetB0': create_transfer_model('EfficientNetB0', config.IMG_SIZE),
}

results = {}

for model_name, model in models_to_train.items():
    print(f"\n{model_name} Summary:")
    # model.summary()

    history, trained_model, metrics = train_model(
        model, model_name, train_gen, val_gen,
        config.EPOCHS, config.LEARNING_RATE
    )

    results[model_name] = metrics
    trained_model.save(f"malaria_{model_name.lower()}.h5")

# 4. Final Comparison Plots
print("\n" + "=" * 50)
print("GENERATING COMPARISON DASHBOARD")
print("=" * 50)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['val_accuracy'] for m in results.keys()],
    'ROC_AUC': [results[m]['roc_auc'] for m in results.keys()]
})

plt.figure(figsize=(10, 6))
plt.bar(comparison_df['Model'], comparison_df['Accuracy'], color=['#764ba2', '#667eea', '#00ff88'])
plt.title('Final Model Accuracy Comparison')
plt.ylabel('Validation Accuracy')
plt.ylim(0, 1.0)
for i, val in enumerate(comparison_df['Accuracy']):
    plt.text(i, val + 0.02, f'{val:.2%}', ha='center', fontweight='bold')

plt.savefig("model_comparison_chart.png")
print("✅ Final comparison chart saved as 'model_comparison_chart.png'")
plt.close()

# Identify the best model
if not comparison_df.empty:
    best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
    print(f"\n🏆 Best Model: {best_model_name}")

    # Production preparation
    src = f"malaria_{best_model_name.lower()}.h5"
    dst = "malaria_model.h5"
    if os.path.exists(src):
        if os.path.exists(dst): os.remove(dst)
        os.rename(src, dst)
        print(f"✅ Best model renamed to '{dst}' for production.")