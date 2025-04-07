# **************************************************************************
# INF7370 Apprentissage automatique
# Travail pratique 2
# ===========================================================================
#Nom et Prenom
# Bellune Tabitha Megane 
# BAGOU A. Ewoenam Gracia

# #===========================================================================
# Ce modèle est un classifieur (un CNN) entraîné sur un ensemble de données
# d’images d’espèces marines afin de distinguer entre six classes :
# baleine, dauphin, morse, phoque, requin, et requin-baleine.
#
# Le jeu de données contient un total de 30 000 images en couleur de taille
# 200x200 pixels, réparties de manière équilibrée entre les six classes.
#
# Données :
# ------------------------------------------------
# Entraînement : 3 000 images par classe → 18 000 images au total
# Validation   : 1 000 images par classe → 6 000 images au total
# Test         : 1 000 images par classe → 6 000 images au total
# ------------------------------------------------
#
# >>> Ce code fonctionne avec un CNN adapté pour traiter ces six classes.
# >>> Vous devez vous assurer que les chemins d’accès aux données,
#     les formats d’images, et les étiquettes correspondent à votre structure.
# >>> Repérez les sections marquées QUESTION pour insérer ou adapter votre code.
# ===========================================================================


# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

from tensorflow import keras

# La libraire responsable du chargement des données dans la mémoire

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Le Type de notre modéle (séquentiel)

from keras.models import Model
from keras.models import Sequential

# Le type d'optimisateur utilisé dans notre modèle (RMSprop, adam, sgd, adaboost ...)
# L'optimisateur ajuste les poids de notre modèle par descente du gradient
# Chaque optimisateur a ses propres paramètres
# Note: Il faut tester plusieurs et ajuster les paramètres afin d'avoir les meilleurs résultats

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau


# Les types des couches utlilisées dans notre modèle
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D

# Des outils pour suivre et gérer l'entrainement de notre modèle
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Sauvegarde du modèle
from keras.models import load_model

# Affichage des graphes
import matplotlib.pyplot as plt


# Affichage du Temps d'exécution
import time

# ==========================================
# ================VARIABLES=================
# ==========================================

# ******************************************************
#                       QUESTION DU TP
# ******************************************************
# 1) Ajuster les variables suivantes selon votre problème:
# - mainDataPath
# - training_batch_size
# - validation_batch_size
# - image_scale
# - image_channels
# - images_color_mode
# - fit_batch_size
# - fit_epochs
# ******************************************************

# Le dossier principal qui contient les données
mainDataPath = "/content/donnees/"

# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "entrainement"

# Le dossier contenant les images de validation
validationPath = mainDataPath + "validation"

# Le dossier contenant les images de test
testPath = mainDataPath + "test"

# Le nom du fichier du modèle à sauvegarder
modelsPath = "/content/donnees/TP2_Model/Model.keras"


# Le nombre d'images d'entrainement et de validation
# Il faut en premier lieu identifier les paramètres du CNN qui permettent d’arriver à des bons résultats. À cette fin, la démarche générale consiste à utiliser une partie des données d’entrainement et valider les résultats avec les données de validation. Les paramètres du réseaux (nombre de couches de convolutions, de pooling, nombre de filtres, etc) devrait etre ajustés en conséquence.  Ce processus devrait se répéter jusqu’au l’obtention d’une configuration (architecture) satisfaisante.
# Si on utilise l’ensemble de données d’entrainement en entier, le processus va être long car on devrait ajuster les paramètres et reprendre le processus sur tout l’ensemble des données d’entrainement.


training_batch_size = 18000  # total 18000 (3000 classe: Dauphin , 3000 classe: phoque, 3000 classe: requin, 3000 classe: morse, 3000 classe: baleine, 3000 classe: Requin-baleine)
validation_batch_size = 6000  # total 6000 (1000 classe: Dauphin , 1000 classe: phoque, 1000 classe: requin, 1000 classe: morse, 1000 classe: baleine, 1000 classe: Requin-baleine)

# Configuration des  images
image_scale = 200 # la taille des images
image_channels = 3  # le nombre de canaux de couleurs (1: pour les images noir et blanc; 3 pour les images en couleurs (rouge vert bleu) )
images_color_mode = "rgb"  # grayscale pour les image noir et blanc; rgb pour les images en couleurs
image_shape = (image_scale, image_scale, image_channels) # la forme des images d'entrées, ce qui correspond à la couche d'entrée du réseau

# Configuration des paramètres d'entrainement
fit_batch_size = 32 # le nombre d'images entrainées ensemble: un batch
fit_epochs = 60 # Le nombre d'époques

# ==========================================
# ==================MODÈLE==================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS DU TP
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Ajuster les deux fonctions:
# 2) feature_extraction
# 3) fully_connected
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Couche d'entrée:
# Cette couche prend comme paramètre la forme des images (image_shape)
input_layer = Input(shape=image_shape)
# Début du chrono
start_time = time.time()

# Partie feature extraction (ou cascade de couches d'extraction des caractéristiques)
def feature_extraction(input):

    # 1-couche de convolution avec nombre de filtre  (exp 32)  avec la taille de la fenetre de ballaiage exp : 3x3
    # 2-fonction d'activation exp: sigmoid, relu, tanh ...
    # 3-couche d'echantillonage (pooling) pour reduire la taille avec la taille de la fenetre de ballaiage exp :2x2

    # **** On répète ces étapes tant que nécessaire ****



    x = Conv2D(32, (3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)



    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # L'ensemble des features/caractéristiques extraits


    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # L'ensemble des features/caractéristiques extraits
    x = Dropout(0.25)(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # L'ensemble des features/caractéristiques extraits
    x = Dropout(0.25)(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)  # L'ensemble des features/caractéristiques extraits


    return encoded


# Partie complètement connectée (Fully Connected Layer)
def fully_connected(encoded):
    # Flatten: pour convertir les matrices en vecteurs pour la couche MLP
    # Dense: une couche neuronale simple avec le nombre de neurone (exemple 64)
    # fonction d'activation exp: sigmoid, relu, tanh ...
    # x = Flatten(input_shape=image_shape)(encoded)
    x = GlobalAveragePooling2D()(encoded)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Activation("relu")(x)

    # Puisque'on a une classification binaire, la dernière couche doit être formée d'un seul neurone avec une fonction d'activation sigmoide
    # La fonction sigmoide nous donne une valeur entre 0 et 1
    # On considère les résultats <=0.5 comme l'image appartenant à la classe 0 (c.-à-d. la classe qui correspond au chiffre 2)
    # on considère les résultats >0.5 comme l'image appartenant à la classe 0 (c.-à-d. la classe qui correspond au chiffre 7)
    x = Dense(6)(x)
    sortie = Activation('softmax')(x)
    return sortie


# Déclaration du modèle:
# La sortie de l'extracteur des features sert comme entrée à la couche complétement connectée
model = Model(input_layer, fully_connected(feature_extraction(input_layer)))

# Affichage des paramétres du modèle
# Cette commande affiche un tableau avec les détails du modèle
# (nombre de couches et de paramétrer ...)
model.summary()

# Compilation du modèle :
# On définit la fonction de perte (exemple :loss='binary_crossentropy' ou loss='mse')
# L'optimisateur utilisé avec ses paramétres (Exemple : optimizer=adam(learning_rate=0.001) )
# La valeur à afficher durant l'entrainement, metrics=['accuracy']
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# ==========================================
# ==========CHARGEMENT DES IMAGES===========
# ==========================================

# training_data_generator: charge les données d'entrainement en mémoire
# quand il charge les images, il les ajuste (change la taille, les dimensions, la direction ...)
# aléatoirement afin de rendre le modèle plus robuste à la position du sujet dans les images
# Note: On peut utiliser cette méthode pour augmenter le nombre d'images d'entrainement (data augmentation)
training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True)

# validation_data_generator: charge les données de validation en memoire
validation_data_generator = ImageDataGenerator(rescale=1. / 255)

# training_generator: indique la méthode de chargement des données d'entrainement
training_generator = training_data_generator.flow_from_directory(
    trainPath, # Place des images d'entrainement
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),# taille des images
    batch_size=training_batch_size, # nombre d'images à entrainer (batch size)
    class_mode="categorical", # classement par categorie (problème de multiclasses)
    shuffle=True) # on "brasse" (shuffle) les données -> pour prévenir le surapprentissage

# validation_generator: indique la méthode de chargement des données de validation
validation_generator = validation_data_generator.flow_from_directory(
    validationPath, # Place des images de validation
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=validation_batch_size,  # nombre d'images à valider
    class_mode="categorical",  # classement par categorie (problème de multiclasses)
    shuffle=True) # on "brasse" (shuffle) les données -> pour prévenir le surapprentissage

# On imprime l'indice de chaque classe (Keras numerote les classes selon l'ordre des dossiers des classes)
# Dans ce cas => [2: 0 et 7:1]
print(training_generator.class_indices)
print(validation_generator.class_indices)

# On charge les données d'entrainement et de validation
# x_train: Les données d'entrainement
# y_train: Les Ètiquettes des données d'entrainement
# x_val: Les données de validation
# y_val: Les Ètiquettes des données de validation
(x_train, y_train) = next(training_generator)
(x_val, y_val) = next(validation_generator)

# ==========================================
# ==============ENTRAINEMENT================
# ==========================================

# Savegarder le modèle avec la meilleure validation accuracy ('val_acc')
# Note: on sauvegarder le modèle seulement quand la précision de la validation s'améliore
modelcheckpoint = ModelCheckpoint(filepath=modelsPath,
                                  monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)


# entrainement du modèle
classifier = model.fit(x_train, y_train,
                       epochs=fit_epochs, # nombre d'époques
                       batch_size=fit_batch_size, # nombre d'images entrainées ensemble
                       validation_data=(x_val, y_val), # données de validation
                       verbose=1, # mets cette valeur ‡ 0, si vous voulez ne pas afficher les détails d'entrainement
                       callbacks=[modelcheckpoint, early_stop, reduce_lr], # les fonctions à appeler à la fin de chaque époque (dans ce cas modelcheckpoint: qui sauvegarde le modèle)
                       shuffle=True)# shuffle les images



# ==========================================
# ========AFFICHAGE DES RESULTATS===========
# ==========================================

# ***********************************************
#                    QUESTION
# ***********************************************
#
# 4) Afficher le temps d'execution
#
# ***********************************************

# Fin du chrono
end_time = time.time()
execution_time = end_time - start_time
print(f"Temps d'exécution de l'entraînement : {execution_time:.2f} secondes")



# ***********************************************
#                    QUESTION
# ***********************************************
#
# 5) Afficher la courbe d’exactitude par époque (Training vs Validation) ainsi que la courbe de perte (loss)
#
# ***********************************************

# Plot accuracy over epochs (precision par époque)
print(classifier.history.keys())
plt.plot(classifier.history['accuracy'])
plt.plot(classifier.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
fig = plt.gcf()
plt.show()

# Courbe de perte (loss)
plt.plot(classifier.history['loss'])
plt.plot(classifier.history['val_loss'])
plt.title('Perte du modèle')
plt.ylabel('Perte')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Validation'])
plt.grid()
plt.show()

