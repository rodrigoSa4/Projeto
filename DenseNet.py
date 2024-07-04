import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import resample
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances

# Função para carregar imagens das pastas correspondentes a diferentes intervalos de tempo
def load_images_from_directories(directories, image_size):
    images = []
    labels = []
    for label, directory in enumerate(directories):
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".png") or filename.endswith(".jpg"):  
                    img_path = os.path.join(root, filename)
                    img = Image.open(img_path).resize(image_size)
                    img = np.array(img) / 255.0  
                    if img.shape == (image_height, image_width, image_channels): 
                        images.append(img)
                        labels.append(label)
    return np.array(images), np.array(labels)

# Configuração de parâmetros
image_height, image_width, image_channels = 128, 128, 3
num_classes = 4  # Número de intervalos de tempo pós impacto

# Pastas de imagens (treino e teste)
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

# Contar o número de exemplos em cada classe pré equilibrio
counter_before = Counter(y_train)
print("Distribuição das classes antes do balanceamento:", counter_before)

# Plotar gráfico de barras com a distribuição de classes antes do equilibrio
plt.figure(figsize=(10, 5))
plt.bar(counter_before.keys(), counter_before.values(), color='blue')
plt.xlabel('Classes')
plt.ylabel('Número de Amostras')
plt.title('Distribuição de Classes Antes do equilibrio')
plt.show()

#  max_count como o número máximo de amostras em qualquer classe
max_count = max(counter_before.values())

# Função para equilibrar as classes
def balance_classes(X_images, y):
    unique_classes = np.unique(y)
    X_images_balanced = []
    y_balanced = []

    for cls in unique_classes:
        class_indices = np.where(y == cls)[0]
        X_images_class = X_images[class_indices]
        y_class = y[class_indices]

        X_images_resampled, y_resampled = resample(
            X_images_class, y_class, 
            replace=True, 
            n_samples=max_count, 
            random_state=42
        )

        X_images_balanced.append(X_images_resampled)
        y_balanced.append(y_resampled)

    return np.vstack(X_images_balanced), np.hstack(y_balanced)

# Aplicar 
X_train_images_balanced, y_train_balanced = balance_classes(X_train_images, y_train)

# Contar o número de exemplos em cada classe depois do equilibrio
counter_after = Counter(y_train_balanced)
print("Distribuição das classes depois do equilibrio:", counter_after)

# Plotar gráfico de barras com a distribuição de classes depois do equilibrio
plt.figure(figsize=(10, 5))
plt.bar(counter_after.keys(), counter_after.values(), color='green')
plt.xlabel('Classes')
plt.ylabel('Número de Amostras')
plt.title('Distribuição de Classes Depois do Equilibrio')
plt.show()

# Função para construir o modelo DenseNet pré-treinado com camadas adicionais e regularização
def create_densenet_model():
    base_model = DenseNet121(
        input_shape=(image_height, image_width, image_channels),
        include_top=False,
        weights='imagenet'
    )

    # Descongelar as últimas camadas
    for layer in base_model.layers[-50:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    image_input = tf.keras.Input(shape=(image_height, image_width, image_channels), name='image_input')
    x = base_model(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=image_input, outputs=x)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# gerador de dados de treino com aumento de dados
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2
)

# Aplicar aumento de dados
train_generator = train_datagen.flow(X_train_images_balanced, y_train_balanced, batch_size=32)

#  pesos de classe
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
class_weights = {i : class_weights[i] for i in range(len(class_weights))}
print("Pesos de classe:", class_weights)

# Treinar o modelo usando Cross-Validation
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

accuracy_per_fold = []
fold_no = 1
history_per_fold = []

for train_index, val_index in skf.split(X_train_images_balanced, y_train_balanced):
    print(f'Treino para o fold {fold_no}...')
    
    X_train_fold, X_val_fold = X_train_images_balanced[train_index], X_train_images_balanced[val_index]
    y_train_fold, y_val_fold = y_train_balanced[train_index], y_train_balanced[val_index]

    model = create_densenet_model()
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train_fold) // 32,
        epochs=30,
        validation_data=(X_val_fold, y_val_fold),
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        ]
    )
    
    scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    accuracy_per_fold.append(scores[1] * 100)
    print(f'Precisão para o fold {fold_no}: {scores[1] * 100}%')
    
    history_per_fold.append(history)
    
    fold_no += 1

# print da precisão média e desvio padrão
print(f'Precisão média: {np.mean(accuracy_per_fold)}%')
print(f'Desvio padrão da precisão: {np.std(accuracy_per_fold)}%')

# Treinar o modelo final em todos os dados de treino equilibrados
final_model = create_densenet_model()

final_model.fit(
    train_generator,
    steps_per_epoch=len(X_train_images_balanced) // 32,
    epochs=30,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.00001)
    ]
)

# Equilibrar o conjunto de teste
X_test_images_balanced, y_test_balanced = balance_classes(X_test_images, y_test)

test_datagen = ImageDataGenerator()

# aumento de dados ao conjunto de teste
test_generator = test_datagen.flow(X_test_images_balanced, y_test_balanced, batch_size=32)

# Avaliar o modelo final no conjunto de teste equilibrado
test_scores = final_model.evaluate(test_generator, verbose=0)
print(f'Precisão no conjunto de teste equilibrado: {test_scores[1]*100:.2f}%')

# Previsões do modelo no conjunto de teste equilibrado
y_pred_balanced = np.argmax(final_model.predict(X_test_images_balanced), axis=1)

# Relatório de classificação no conjunto de teste equilibrado
print("Relatório de Classificação:")
print(classification_report(y_test_balanced, y_pred_balanced))

# Matriz de confusão 
cm = confusion_matrix(y_test_balanced, y_pred_balanced)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.title("Matriz de Confusão (Conjunto de Teste equilibrado)")
plt.show()

# Curva ROC
y_test_bin = label_binarize(y_test_balanced, classes=[0, 1, 2, 3])
y_pred_prob = final_model.predict(X_test_images_balanced)
fpr, tpr, roc_auc = {}, {}, {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Classe {i} (área = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curvas ROC Multiclasse (Conjunto de Teste Equilibrado)')
plt.legend(loc="lower right")
plt.show()
