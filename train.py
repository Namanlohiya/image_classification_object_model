from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# 1. Correct Path Handling (Windows compatible)
BASE_DIR = 'image_classifier'
DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'efficientnet_model.h5')

# 2. Verify dataset structure
if not os.path.exists(DATA_DIR):
    print(f"Error: Dataset directory not found at {os.path.abspath(DATA_DIR)}")
    print("Please create this structure:")
    print(f"{BASE_DIR}/dataset/train/")
    print("├── cat/")
    print("├── dog/")
    print("└── bird/")
    exit()

# 3. Single Data Generator Definition (removed duplicate)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 4. Create model using transfer learning
base_model = MobileNetV2(weights='imagenet', 
                        include_top=False, 
                        input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(3, activation='softmax')(x)  # 3 classes
model = Model(inputs=base_model.input, outputs=predictions)

# 5. Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Training with consistent paths
train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 7. Train model
print("\nStarting training...")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20
)

# 8. Save model
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")