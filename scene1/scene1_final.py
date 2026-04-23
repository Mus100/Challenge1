# scene1_final.py
# Segmentation de la Scène 1 (chat, ciel, sol, arbres)
# Basé sur : TP1, TP2, TP3, TP4, TP6
# Auteur : Personne 1

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans  # utilisé uniquement pour lire le GT

os.makedirs("results/scene1", exist_ok=True)

# ══════════════════════════════════
# ÉTAPE 1 : CHARGEMENT
# ══════════════════════════════════

img_bgr = cv2.imread("data/scene1/Scene_1.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W, _ = img_rgb.shape
print(f"Image chargée : {H}x{W}")

# ══════════════════════════════════
# ÉTAPE 2 : PRÉTRAITEMENT (TP4 + TP2)
# ══════════════════════════════════

# --- Filtre Gaussien FROM SCRATCH (TP4) ---
def filtre_gaussien_noyau(taille=3, sigma=1.0):
    k = taille // 2
    noyau = np.zeros((taille, taille))
    for x in range(-k, k+1):
        for y in range(-k, k+1):
            noyau[x+k, y+k] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return noyau / noyau.sum()

def convolution_canal(canal, noyau):
    H, W = canal.shape
    N = noyau.shape[0]
    pad = N // 2
    img_pad = np.pad(canal, pad, mode='edge').astype(np.float64)
    res = np.zeros_like(canal, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            res[i, j] = np.sum(img_pad[i:i+N, j:j+N] * noyau)
    return np.clip(res, 0, 255).astype(np.uint8)

print(" Application du filtre gaussien (TP4)...")
noyau_g = filtre_gaussien_noyau(taille=3, sigma=1.0)

img_filtre = np.zeros_like(img_rgb)
for c in range(3):
    img_filtre[:, :, c] = convolution_canal(img_rgb[:, :, c], noyau_g)

print(" Filtre gaussien appliqué")

# --- Conversion en espace LAB ---
img_lab = cv2.cvtColor(img_filtre, cv2.COLOR_RGB2LAB)
print("Conversion LAB effectuée")

# --- Visualisation prétraitement ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_rgb);    axes[0].set_title("1 - Image Originale")
axes[1].imshow(img_filtre); axes[1].set_title("2 - Après Filtre Gaussien (TP4)")
axes[2].imshow(img_lab);    axes[2].set_title("3 - Espace LAB")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig("results/scene1/etape2_pretraitement.png")
plt.show()

# ══════════════════════════════════
# ÉTAPE 3 : K-MEANS FROM SCRATCH (TP3)
# ══════════════════════════════════

K = 4  # chat, ciel, sol, arbres

pixels = img_lab.reshape(-1, 3).astype(np.float64)
mu  = pixels.mean(axis=0)
std = pixels.std(axis=0) + 1e-8
pixels_norm = (pixels - mu) / std

print(f"\n Lancement K-Means K={K}...")
print(f"   Nombre de pixels : {len(pixels_norm)}")

# Étape a) Initialisation
np.random.seed(42)
indices_init = np.random.choice(len(pixels_norm), K, replace=False)
centres = pixels_norm[indices_init].copy()
print(" Centres initialisés")

# Boucle K-Means
for iteration in range(50):

    # Étape b) Assignation
    distances = np.sqrt(
        ((pixels_norm[:, np.newaxis, :] - centres[np.newaxis, :, :]) ** 2
        ).sum(axis=2)
    )
    labels = np.argmin(distances, axis=1)

    # Étape c) Mise à jour des centres
    nouveaux_centres = np.zeros_like(centres)
    for k in range(K):
        membres = pixels_norm[labels == k]
        if len(membres) > 0:
            nouveaux_centres[k] = membres.mean(axis=0)
        else:
            nouveaux_centres[k] = centres[k]

    # Étape d) Vérifier convergence
    deplacement = np.sqrt(((nouveaux_centres - centres) ** 2).sum())
    centres = nouveaux_centres
    print(f"   Itération {iteration+1:02d} - déplacement : {deplacement:.6f}")

    if deplacement < 1e-4:
        print(f" Convergence à l'itération {iteration+1} !")
        break

labels_2d = labels.reshape(H, W)
print(" K-Means terminé !")

# Visualisation clusters bruts
couleurs_visu = np.array([
    [255,   0,   0],
    [  0, 200,   0],
    [  0,   0, 255],
    [255, 200,   0],
], dtype=np.uint8)

img_clusters = couleurs_visu[labels_2d]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img_rgb);      axes[0].set_title("Image Originale")
axes[1].imshow(img_clusters); axes[1].set_title("Clusters K-Means brut (TP3)")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig("results/scene1/etape3_kmeans.png")
plt.show()

print("\n Taille des clusters :")
for k in range(K):
    taille = np.sum(labels == k)
    pct = taille / len(labels) * 100
    print(f"   Cluster {k} : {taille} pixels ({pct:.1f}%)")

# ══════════════════════════════════
# ÉTAPE 4 : IDENTIFICATION DES CLUSTERS
# ══════════════════════════════════

centres_lab_reels = np.zeros((K, 3))
for k in range(K):
    centres_lab_reels[k] = pixels[labels == k].mean(axis=0)

print("\n Centres des clusters (espace LAB réel) :")
for k in range(K):
    L, A, B = centres_lab_reels[k]
    print(f"   Cluster {k} : L={L:.1f}  A={A:.1f}  B={B:.1f}")

# Assignation manuelle basée sur le diagnostic
# Cluster 0 : L=235.5 -> très lumineux = CIEL
# Cluster 1 : L=113.6 -> bleu-vert     = ARBRES
# Cluster 2 : L=87.5  -> sombre        = CHAT
# Cluster 3 : L=134.7 -> vert clair    = SOL
noms = {
    0 : "ciel",
    1 : "arbres",
    2 : "chat",
    3 : "sol"
}

print("\n  Assignation des clusters :")
for k, nom in noms.items():
    L, A, B = centres_lab_reels[k]
    print(f"   Cluster {k} -> {nom}  (L={L:.1f}, A={A:.1f}, B={B:.1f})")

# Visualisation avec noms
couleurs_noms = {
    "chat"   : [255, 220, 130],
    "ciel"   : [ 80, 140, 210],
    "sol"    : [100, 180, 100],
    "arbres" : [ 40, 100,  40],
}

img_nommee = np.zeros((H, W, 3), dtype=np.uint8)
for k, nom in noms.items():
    img_nommee[labels_2d == k] = couleurs_noms[nom]

# ══════════════════════════════════
# ÉTAPE 5 : POST-TRAITEMENT MORPHOLOGIQUE (TP6)
# ══════════════════════════════════

print("\n Post-traitement morphologique...")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
labels_nettoyes = labels_2d.copy()

for k in range(K):
    nom = noms[k]
    masque = (labels_2d == k).astype(np.uint8)
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, kernel)
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN,  kernel)
    labels_nettoyes[masque == 1] = k
    print(f"    {nom} nettoyé")

img_nettoyee = np.zeros((H, W, 3), dtype=np.uint8)
for k, nom in noms.items():
    img_nettoyee[labels_nettoyes == k] = couleurs_noms[nom]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(img_rgb);    axes[0].set_title("Image Originale")
axes[1].imshow(img_nommee); axes[1].set_title("Avant nettoyage")
axes[2].imshow(img_nettoyee); axes[2].set_title("Après nettoyage morphologique (TP6)")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig("results/scene1/etape5_morphologie.png")
plt.show()

print("\n Comparaison avant/après nettoyage :")
for k in range(K):
    avant = np.sum(labels_2d == k)
    apres = np.sum(labels_nettoyes == k)
    diff  = apres - avant
    signe = "+" if diff >= 0 else ""
    print(f"   {noms[k]:10s} : {avant} -> {apres} pixels  ({signe}{diff})")

# ══════════════════════════════════
# ÉTAPE 6 : ÉVALUATION AVEC GT1
# ══════════════════════════════════

print("\n Évaluation des résultats...")

gt_bgr = cv2.imread("data/scene1/GT1.png")
gt_bgr = cv2.resize(gt_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
print(f"   Taille Scene_1 : {H}x{W}")
print(f"   Taille GT1     : {gt_bgr.shape[0]}x{gt_bgr.shape[1]}")

# K-Means sklearn uniquement pour extraire les 4 couleurs du GT
pixels_gt = gt_rgb.reshape(-1, 3).astype(np.float32)
km_gt = KMeans(n_clusters=4, random_state=42, n_init=10)
km_gt.fit(pixels_gt)
centres_gt = km_gt.cluster_centers_.astype(int)
labels_gt  = km_gt.labels_.reshape(H, W)

# Assignation automatique des couleurs GT
assignation_gt = {}
for i, c in enumerate(centres_gt):
    R, G, B = c
    if B > R and B > G:
        assignation_gt[i] = "ciel"
    elif R > 200 and G > 190 and B < 170:
        assignation_gt[i] = "chat"
    elif G > R and G > B:
        assignation_gt[i] = "sol"
    else:
        assignation_gt[i] = "arbres"

# Construire les masques GT
masques_gt = {}
for i, nom in assignation_gt.items():
    masques_gt[nom] = (labels_gt == i).astype(np.uint8)

# Fonction métriques
def calculer_metriques(masque_gt, masque_pred, nom=""):
    gt   = masque_gt.flatten().astype(bool)
    pred = masque_pred.flatten().astype(bool)
    TP = np.sum( gt &  pred)
    FP = np.sum(~gt &  pred)
    FN = np.sum( gt & ~pred)
    TN = np.sum(~gt & ~pred)
    iou       = TP / (TP + FP + FN + 1e-8)
    dice      = 2*TP / (2*TP + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    rappel    = TP / (TP + FN + 1e-8)
    f1        = 2*precision*rappel / (precision + rappel + 1e-8)
    pixel_acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    return {
        "Classe"        : nom,
        "IoU"           : round(float(iou),       4),
        "Dice"          : round(float(dice),       4),
        "Précision"     : round(float(precision),  4),
        "Rappel"        : round(float(rappel),     4),
        "F1-score"      : round(float(f1),         4),
        "Pixel Accuracy": round(float(pixel_acc),  4),
    }

# Calcul des métriques
tous_resultats = []
for nom, masque_gt_i in masques_gt.items():
    k_id = [k for k, v in noms.items() if v == nom]
    if k_id:
        masque_pred = (labels_nettoyes == k_id[0]).astype(np.uint8)
        m = calculer_metriques(masque_gt_i, masque_pred, nom)
        tous_resultats.append(m)

# Afficher le tableau
print("\n" + "="*75)
print(f"  RÉSULTATS SCÈNE 1")
print("="*75)
print(f"  {'Classe':<12} {'IoU':>8} {'Dice':>8} "
      f"{'Précision':>10} {'Rappel':>8} {'F1':>8} {'PixAcc':>8}")
print("  " + "-"*70)
for m in tous_resultats:
    print(f"  {m['Classe']:<12} "
          f"{m['IoU']:>8} "
          f"{m['Dice']:>8} "
          f"{m['Précision']:>10} "
          f"{m['Rappel']:>8} "
          f"{m['F1-score']:>8} "
          f"{m['Pixel Accuracy']:>8}")
print("="*75)

iou_moy  = np.mean([m['IoU']            for m in tous_resultats])
dice_moy = np.mean([m['Dice']           for m in tous_resultats])
f1_moy   = np.mean([m['F1-score']       for m in tous_resultats])
acc_moy  = np.mean([m['Pixel Accuracy'] for m in tous_resultats])

print(f"\n  {'MOYENNE':<12} "
      f"{iou_moy:>8.4f} "
      f"{dice_moy:>8.4f} "
      f"{'':>10} "
      f"{'':>8} "
      f"{f1_moy:>8.4f} "
      f"{acc_moy:>8.4f}")
print("="*75)

# Visualisation GT vs Prédits
classes = ["chat", "ciel", "sol", "arbres"]
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for i, nom in enumerate(classes):
    k_id = [k for k, v in noms.items() if v == nom]
    if nom in masques_gt:
        axes[0, i].imshow(masques_gt[nom], cmap='gray')
    else:
        axes[0, i].imshow(np.zeros((H, W)), cmap='gray')
    axes[0, i].set_title(f"GT - {nom}")
    axes[0, i].axis('off')
    if k_id:
        masque_pred = (labels_nettoyes == k_id[0]).astype(np.uint8)
        axes[1, i].imshow(masque_pred, cmap='gray')
    else:
        axes[1, i].imshow(np.zeros((H, W)), cmap='gray')
    axes[1, i].set_title(f"Prédit - {nom}")
    axes[1, i].axis('off')

plt.suptitle(
    f"Scène 1 — IoU moy={iou_moy:.3f} | "
    f"Dice moy={dice_moy:.3f} | "
    f"F1 moy={f1_moy:.3f}",
    fontsize=13
)
plt.tight_layout()
plt.savefig("results/scene1/etape6_evaluation.png")
plt.show()

# ══════════════════════════════════
# ÉTAPE 7 : SAUVEGARDE FINALE
# ══════════════════════════════════

print("\n Sauvegarde des résultats finaux...")

# --- 1. Image finale de segmentation ---
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Scene_1 — Image originale", fontsize=12)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_nettoyee)
for nom, couleur in couleurs_noms.items():
    plt.plot([], [], 's',
             color=np.array(couleur)/255,
             label=nom, markersize=12)
plt.legend(loc='lower right', fontsize=9)
plt.title("Segmentation K-Means (from scratch)", fontsize=12)
plt.axis('off')

plt.subplot(1, 3, 3)
gt_img = cv2.cvtColor(cv2.imread("data/scene1/GT1.png"), cv2.COLOR_BGR2RGB)
plt.imshow(gt_img)
plt.title("Ground Truth GT1", fontsize=12)
plt.axis('off')

plt.suptitle(
    f"Scène 1 — IoU={iou_moy:.3f} | Dice={dice_moy:.3f} | F1={f1_moy:.3f}",
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig("results/scene1/resultat_final.png", dpi=150, bbox_inches='tight')
plt.show()
print("results/scene1/resultat_final.png")

# --- 2. Masques binaires de chaque classe ---
for k, nom in noms.items():
    masque = (labels_nettoyes == k).astype(np.uint8) * 255
    chemin = f"results/scene1/masque_{nom}.png"
    cv2.imwrite(chemin, masque)
    print(f" {chemin}")

# --- 3. Tableau métriques en image ---
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

entetes = ["Classe", "IoU", "Dice", "Précision", "Rappel", "F1", "Pixel Acc"]
donnees = [
    [m["Classe"], m["IoU"], m["Dice"],
     m["Précision"], m["Rappel"],
     m["F1-score"], m["Pixel Accuracy"]]
    for m in tous_resultats
]
donnees.append([
    "MOYENNE",
    round(iou_moy, 4), round(dice_moy, 4),
    "-", "-",
    round(f1_moy, 4), round(acc_moy, 4)
])

table = ax.table(
    cellText=donnees,
    colLabels=entetes,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Colorier l'en-tête
for j in range(len(entetes)):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Colorier la ligne MOYENNE
for j in range(len(entetes)):
    table[len(donnees), j].set_facecolor('#ecf0f1')
    table[len(donnees), j].set_text_props(fontweight='bold')

plt.title("Tableau des métriques — Scène 1", fontsize=14,
          fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("results/scene1/tableau_metriques.png",
            dpi=150, bbox_inches='tight')
plt.show()
print(" results/scene1/tableau_metriques.png")

# --- 4. Résumé final dans le terminal ---
print("\n" + "="*55)
print("  RÉSUMÉ FINAL — SCÈNE 1")
print("="*55)
print(f"  Algorithme    : K-Means from scratch (K=4)")
print(f"  Prétraitement : Filtre Gaussien (TP4)")
print(f"  Post-trait.   : Morphologie (TP6)")
print(f"  IoU moyen     : {iou_moy:.4f}")
print(f"  Dice moyen    : {dice_moy:.4f}")
print(f"  F1 moyen      : {f1_moy:.4f}")
print(f"  Pixel Acc.    : {acc_moy:.4f}")
print("="*55)
print("\n  Fichiers sauvegardés dans results/scene1/ :")
print("    - etape2_pretraitement.png")
print("    - etape3_kmeans.png")
print("    - etape5_morphologie.png")
print("    - etape6_evaluation.png")
print("    - resultat_final.png")
print("    - tableau_metriques.png")
print("    - masque_chat.png")
print("    - masque_ciel.png")
print("    - masque_sol.png")
print("    - masque_arbres.png")
print("="*55)
print("\n Scène 1 entièrement terminée !")
