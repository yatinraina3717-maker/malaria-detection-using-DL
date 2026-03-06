# MALARIA DETECTION - MODEL BUILDING
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam

# CONFIGURATION

IMG_SHAPE   = (128, 128, 3)    
NUM_CLASSES = 2                 # Parasitized, Uninfected

# MODEL 1: CUSTOM CNN

def create_custom_cnn(img_shape=IMG_SHAPE, num_classes=NUM_CLASSES):

    model = Sequential(name="CustomCNN")

    # Block 1 — output: 64x64x32
    model.add(Conv2D(32, (3, 3), activation='relu',
                     padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    # Block 2 — output: 32x32x64
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    # Block 3 — output: 16x16x128
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    # Block 4 — output: 8x8x256
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# MODEL 2 & 3: TRANSFER LEARNING

def create_transfer_model(model_name, img_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    
    if model_name == 'MobileNetV2':
        base_model = MobileNetV2(
            input_shape=img_shape,
            include_top=False,
            weights='imagenet'
        )

    elif model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(
            input_shape=img_shape,
            include_top=False,
            weights='imagenet'
        )

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Choose 'MobileNetV2' or 'EfficientNetB0'"
        )

    # Freeze base model layers
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # final model
    model = Model(
        inputs=base_model.input,
        outputs=output,
        name=model_name
    )

    # Compile with lower learning rate for transfer learning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# VERIFY — python task2_model_building.py

if __name__ == "__main__":

    print("\n" + "="*50)
    print("TASK 2 — MODEL BUILDING")
    print("="*50)

    # Build all 3 models
    cnn            = create_custom_cnn()
    mobilenet      = create_transfer_model('MobileNetV2')
    efficientnet   = create_transfer_model('EfficientNetB0')

    # Print summaries
    print("\n── CustomCNN ──")
    cnn.summary()

    print("\n── MobileNetV2 ──")
    mobilenet.summary()

    print("\n── EfficientNetB0 ──")
    efficientnet.summary()

    print("\n✅ All 3 models built successfully")
    print("✅ Ready to hand over to Task 3")