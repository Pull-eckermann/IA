import numpy as np
import tensorflow as tf
import keras
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.applications import VGG16, ResNet50, ResNet50V2, EfficientNetB4
from keras.applications.resnet import preprocess_input
from matplotlib import pyplot as plt
import pandas as pd

tf.get_logger().setLevel('ERROR')

# Tamanho imagem
img_rows, img_cols = 224, 224

input_shape = (img_rows, img_cols, 3)

#Batch size

batch_size = 32

#Apresentando base de dados
path_train = './simpsons/Treino/'
path_test = './simpsons/Valid/'
path_train_c = './simpsons_cropped/20_simpsons_crop/Treino/'
path_test_c = './simpsons_cropped/20_simpsons_crop/Valid/'


train_dataset = tf.keras.utils.image_dataset_from_directory(
        path_train_c,
        image_size=(img_rows, img_cols),
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False)
test_dataset = tf.keras.utils.image_dataset_from_directory(
        path_test_c,
        image_size=(img_rows, img_cols),
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False)

AUTOTUNE = tf.data.AUTOTUNE
  
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE) #Otimização
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

#============================TREINANDO BASE======================
# Parâmetros
num_classes = 6

n_epochs = 100

seed = 42

patience = 15

# loss
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# métricas
metrics = ['accuracy']  

#Data augmentation
data_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                         tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
                                         tf.keras.layers.experimental.preprocessing.RandomContrast(0.2, 0.2)])

#Criando modelo extrator de características
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape) #Corta camadas
resnet50.trainable = False #Para o primeiro treinamento as camadas convolucionais não serão treinadas
inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = resnet50(x, training=False) #Camadas de Batch Normalization em inference mode
x = GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.4)(x) # Camada de Dropout
predictions = Dense(num_classes)(x) #Camada Densa sem ativação, pois a loss foi definida como from_logits=True
model = Model(inputs, predictions)     

#Treinamento camada densa
opt_dense = keras.optimizers.Adam()
model.compile(metrics=metrics, loss=loss, optimizer=opt_dense)
callbacks_dense = [EarlyStopping(patience=patience)]
history_dense =model.fit(train_dataset, epochs=n_epochs, verbose=1, validation_data=test_dataset, callbacks=callbacks_dense)

#=====================VETOR DE CARACTERISTICAS====================
#Extraindo características
X_train = model.predict(train_dataset)
X_test = model.predict(test_dataset)
y_train = np.concatenate([y for x, y in train_dataset], axis=0)
y_test = np.concatenate([y for x, y in test_dataset], axis=0)

#==============================RESULTS=============================
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing

# cria um kNN
neigh = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
print ('Fitting knn')
neigh.fit(X_train, y_train)

# predicao do classificador
print ('Predicting...')
y_pred = neigh.predict(X_test)

# mostra o resultado do classificador na base de teste
print ('Accuracy: ',  neigh.score(X_test, y_test))

# cria a matriz de confusao
cm = confusion_matrix(y_test, y_pred)
print (cm)
print(classification_report(y_test, y_pred))

#Gera gráfico
pdhistory = pd.DataFrame(history_dense.history)
pdhistory.index += 1
plt.figure(figsize=(20,8))
plt.plot(pdhistory['loss'])
plt.plot(pdhistory['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(str(batch_size) + "_" + str(patience) + "_" + 'simpsons_cropped__loss' + ".png")

plt.figure(figsize=(20,8))
plt.plot(pdhistory['accuracy'])
plt.plot(pdhistory['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(str(batch_size) + "_" + str(patience) + "_" + 'simpsons_cropped_accuracy' + ".png")