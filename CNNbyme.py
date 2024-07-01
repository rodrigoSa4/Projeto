import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import resample
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

# Função para carregar imagens das pastas correspondentes a diferentes intervalos de tempo
def load_images_from_directories(directories, image_size):
    images = []
    labels = []
    for label, directory in enumerate(directories):
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".png") or filename.endswith(".jpg"):  # Ajuste conforme necessário
                    img_path = os.path.join(root, filename)
                    img = Image.open(img_path).resize(image_size)
                    img = np.array(img) / 255.0  # Normalizar a imagem para [0, 1]
                    if img.shape == (image_height, image_width, image_channels):  # Certifique-se de que a imagem tem o formato correto
                        images.append(img)
                        labels.append(label)
    return np.array(images), np.array(labels)

# Configuração de parâmetros
image_height, image_width, image_channels = 128, 128, 3
num_classes = 4  # Número de intervalos de tempo pós impacto

#  pastas de imagens (treino e teste)
train_directories = [
    "F:\\UA\\3 ano\\2 semestre\\Projeto de licenciatura\\Codigo\\TBI-AD Microscope Images\\TBI-AD Microscope Images\\training\\1 day",
    "F:\\UA\\3 ano\\2 semestre\\Projeto de licenciatura\\Codigo\\TBI-AD Microscope Images\\TBI-AD Microscope Images\\training\\1 week",
    "F:\\UA\\3 ano\\2 semestre\\Projeto de licenciatura\\Codigo\\TBI-AD Microscope Images\\TBI-AD Microscope Images\\training\\1 month",
    "F:\\UA\\3 ano\\2 semestre\\Projeto de licenciatura\\Codigo\\TBI-AD Microscope Images\\TBI-AD Microscope Images\\training\\3 months"
]

test_directories = [
    "F:\\UA\\3 ano\\2 semestre\\Projeto de licenciatura\\Codigo\\TBI-AD Microscope Images\\TBI-AD Microscope Images\\test\\1 day",
    "F:\\UA\\3 ano\\2 semestre\\Projeto de licenciatura\\Codigo\\TBI-AD Microscope Images\\TBI-AD Microscope Images\\test\\1 week",
    "F:\\UA\\3 ano\\2 semestre\\Projeto de licenciatura\\Codigo\\TBI-AD Microscope Images\\TBI-AD Microscope Images\\test\\1 month",
    "F:\\UA\\3 ano\\2 semestre\\Projeto de licenciatura\\Codigo\\TBI-AD Microscope Images\\TBI-AD Microscope Images\\test\\3 months"
]

# Carregar e pré-processar imagens
X_train_images, y_train = load_images_from_directories(train_directories, (image_height, image_width))
print(f"Número de amostras de treino: {len(X_train_images)}")
print(f"Formas das imagens de treino: {X_train_images.shape}")
print(f"Etiquetas de treino: {y_train}")

# Carregar as imagens de teste
X_test_images, y_test = load_images_from_directories(test_directories, (image_height, image_width))
print(f"Número de amostras de teste: {len(X_test_images)}")
print(f"Formas das imagens de teste: {X_test_images.shape}")
print(f"Etiquetas de teste: {y_test}")

# Contar o número de amostras em cada classe
counter = Counter(y_train)
print(counter)

# Equilibrio de classes
class_counts = Counter(y_train)
max_count = max(class_counts.values())

# Função para equilibrar as classes
def balance_classes(X_images, y):
    unique_classes = np.unique(y)
    X_images_balanced = []
    y_balanced = []

    for cls in unique_classes:
        class_indices = np.where(y == cls)[0]
        class_indices = tf.convert_to_tensor(class_indices, dtype=tf.int32)  # Converter para tensor de int32
        X_images_class = tf.gather(X_images, class_indices).numpy()
        y_class = tf.gather(y, class_indices).numpy()

        X_images_resampled, y_resampled = resample(
            X_images_class, y_class, 
            replace=True, 
            n_samples=max_count, 
            random_state=42
        )

        X_images_balanced.append(X_images_resampled)
        y_balanced.append(y_resampled)

    return np.vstack(X_images_balanced), np.hstack(y_balanced)

# Aplicar o balanceamento
X_train_images_balanced, y_train_balanced = balance_classes(X_train_images, y_train)

print("Número de amostras de treino após balanceamento:", len(y_train_balanced))
print("Distribuição das classes no conjunto de treino balanceado:", Counter(y_train_balanced))

# Função para construir o modelo 
def create_model():
    image_input = tf.keras.Input(shape=(image_height, image_width, image_channels), name='image_input')

    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=image_input, outputs=output)

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Configuração da validação cruzada
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
validation_scores = []

for train_index, val_index in kf.split(X_train_images_balanced, y_train_balanced):
    X_train_fold_images, X_val_fold_images = X_train_images_balanced[train_index], X_train_images_balanced[val_index]
    y_train_fold, y_val_fold = y_train_balanced[train_index], y_train_balanced[val_index]

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    model = create_model()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    model.fit(
        datagen.flow(X_train_fold_images, y_train_fold, batch_size=32),
        validation_data=(X_val_fold_images, y_val_fold),
        epochs=100,
        callbacks=[early_stopping, reduce_lr]
    )
    
    val_score = model.evaluate(X_val_fold_images, y_val_fold, verbose=0)
    validation_scores.append(val_score[1])
    print(f"Validação Fold Score: {val_score[1]}")

print(f"Scores de Validação Cruzada: {validation_scores}")
print(f"Score Médio de Validação Cruzada: {np.mean(validation_scores)}")

# Treino final com todos os dados de treino
model = create_model()
history = model.fit(
    X_train_images_balanced, y_train_balanced,
    validation_data=(X_test_images, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)

# Avaliação do modelo
test_loss, test_accuracy = model.evaluate(X_test_images, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plotar gráfico de scores de validação por split
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(validation_scores) + 1), validation_scores, marker='o', linestyle='-', label='Validation Score por Split')
plt.xlabel('Fold')
plt.ylabel('Validation Score')
plt.title('Validation Scores por Split na Validação Cruzada')
plt.legend()
plt.show()

# matriz de confusao
y_pred = np.argmax(model.predict(X_test_images), axis=-1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Plotar 5 imagens aleatórias do conjunto de teste com as classes reais e previstas
num_images = 5
random_indices = np.random.choice(len(X_test_images), num_images, replace=False)
sample_images = X_test_images[random_indices]
sample_labels = y_test[random_indices]
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(15, 15))
for i in range(num_images):
    plt.subplot(5, 2, i + 1)
    plt.imshow(sample_images[i])
    plt.title(f"True: {sample_labels[i]}, Predicted: {predicted_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Plotar curva ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
y_pred_prob = model.predict(X_test_images)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 7))
colors = ['blue', 'green', 'red', 'orange']
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


