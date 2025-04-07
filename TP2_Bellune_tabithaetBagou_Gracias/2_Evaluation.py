# **************************************************************************
# INF7370 Apprentissage automatique
# Travail pratique 2
# ===========================================================================
#Nom et Prenom
# Bellune Tabitha Megane 
# BAGOU A. Ewoenam Gracia
#===========================================================================
# Dans ce script, on évalue le modèle entrainé dans 1_Modele.py
# On charge le modèle en mémoire; on charge les images; et puis on applique le modèle sur les images afin de prédire les classes



# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

from tensorflow import keras

# La libraire responsable du chargement des données dans la mémoire
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes
import matplotlib.pyplot as plt

# La librairie numpy
import numpy as np

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Utilisé pour le calcul des métriques de validation
from sklearn.metrics import confusion_matrix, roc_curve , auc

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from PIL import Image

# ==========================================
# ==================MODÈLE==================
# ==========================================

#Chargement du modéle sauvegardé dans la section 1 via 1_Modele.py
model_path =  "/content/donnees/TP2_Model/Model.keras"
Classifier: Model = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1) A ajuster les variables suivantes selon votre problème:
# - mainDataPath
# - number_images
# - number_images_class_x
# - image_scale
# - images_color_mode
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# L'emplacement des images de test
mainDataPath = "/content/donnees/"
testPath = mainDataPath + "test"


# Le nombre des images de test à évaluer
number_images = 6000 # 1000 images par classe (1000 classe: Dauphin , 1000 classe: phoque, 1000 classe: requin, 1000 classe: morse, 1000 classe: baleine, 1000 classe: Requin-baleine)
number_images_class_0 = 1000
number_images_class_1 = 1000
number_images_class_2 = 1000
number_images_class_3 = 1000
number_images_class_4 = 1000
number_images_class_5 = 1000

# La taille des images à classer
image_scale = 200

# La couleur des images à classer
images_color_mode = "rgb"  # grayscale or rgb

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images de test
test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_itr = test_data_generator.flow_from_directory(
    testPath,# place des images
    target_size=(image_scale, image_scale), # taille des images
    class_mode="categorical",# Type de classification
    shuffle=False,# pas besoin de les boulverser
    batch_size=1,# on classe les images une à la fois
    color_mode=images_color_mode)# couleur des images

(x, y_true) = next(test_itr)

# ==========================================
# ===============ÉVALUATION=================
# ==========================================

# Les classes correctes des images (1000 pour chaque classe) -- the ground truth
y_true = np.array([0] * number_images_class_0 +
                  [1] * number_images_class_1 +
                  [2] * number_images_class_2 +
                  [3] * number_images_class_3 +
                  [4] * number_images_class_4 +
                  [5] * number_images_class_5
                  )

# evaluation du modËle
test_eval = Classifier.evaluate(test_itr, verbose=1)

# Affichage des valeurs de perte et de precision
print('>Test loss (Erreur):', test_eval[0])
print('>Test précision:', test_eval[1])

# Prédiction des classes des images de test
predicted_classes = Classifier.predict(test_itr, verbose=1)
predicted_classes_perc = np.round(predicted_classes.copy(), 4)
# (0: Dauphin , 1: phoque, 2: requin, 3: morse, 4: baleine, 5: Requin-baleine)

# Transformation en labels entiers
predicted_labels = np.argmax(predicted_classes, axis=1)

# Comparaison avec les vraies classes ( etiquettes bien classees et mal classees)
correct = []
incorrect = []

for i in range(len(predicted_labels)):
    if predicted_labels[i] == y_true[i]:
        correct.append(i)
    else:
        incorrect.append(i)

# Affichage
print("> %d étiquettes bien classées" % len(correct))
print("> %d étiquettes mal classées" % len(incorrect))

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 1) Afficher la matrice de confusion

 # matrice de confusion
cm = confusion_matrix(y_true, predicted_labels)
class_names = list(test_itr.class_indices.keys())  # noms de classes à partir des dossiers

# affichage
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Prédiction")
plt.ylabel("Réelle")
plt.title("Matrice de confusion")
plt.savefig("/content/confusion.png", dpi=200)
plt.show()

# 2) Extraire une image mal-classée pour chaque combinaison d'espèces - Voir l'exemple dans l'énoncé.
wrong_examples = {}
filenames = test_itr.filenames  # chemins des fichiers image
image_dir = testPath            # dossier racine

for i in range(len(predicted_labels)):
    true_label = y_true[i]
    pred_label = predicted_labels[i]
    if true_label != pred_label:
        key = (true_label, pred_label)
        if key not in wrong_examples:
            wrong_examples[key] = i  # on stocke un seul exemple par combinaison


fig, axes = plt.subplots(6, 6, figsize=(15, 15))
fig.suptitle("Images mal classées : Réel (lignes) vs Prédit (colonnes)", fontsize=16)

for i in range(6):  # Réel
    for j in range(6):  # Prédit
        ax = axes[i, j]
        ax.axis("off")

        if i == j:
            ax.set_facecolor('lightgrey')
            ax.text(0.5, 0.5, "✓", ha='center', va='center', fontsize=20, color='white')
        else:
            key = (i, j)
            if key in wrong_examples:
                idx = wrong_examples[key]
                img_path = os.path.join(image_dir, filenames[idx])
                try:
                    img = Image.open(img_path).resize((100, 100))
                    ax.imshow(img)
                except:
                    ax.set_facecolor('red')
                    ax.text(0.5, 0.5, "Erreur", ha='center', va='center', color='white')
            else:
                ax.set_facecolor('#f2f2f2')
                ax.text(0.5, 0.5, "Aucune", ha='center', va='center', fontsize=8, color='gray')

        if i == 0:
            ax.set_title(class_names[j], fontsize=12)

# Ajouter manuellement les labels à gauche de chaque ligne
for i, label in enumerate(class_names):
    fig.text(0.07, 0.80 - i * (0.128), label, va='center', ha='right', fontsize=12, linespacing=0.9)

plt.tight_layout()
plt.subplots_adjust(top=0.92, left=0.12)  # Ajustement pour laisser de la place à gauche
plt.savefig("/content/mal_classe.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ***********************************************
