# ============================================================
# utils.py
# Fonctions utilitaires communes à tout le projet

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ══════════════════════════════════════════════
# 1. CHARGEMENT DES IMAGES
# ══════════════════════════════════════════════

def charger_image(chemin):
    """
    Charge une image couleur.
    Retourne : img_bgr, img_rgb
    """
    img_bgr = cv2.imread(chemin)
    if img_bgr is None:
        raise FileNotFoundError(f"Image introuvable : {chemin}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image chargée : {img_rgb.shape[0]}x{img_rgb.shape[1]} — {chemin}")
    return img_bgr, img_rgb

def charger_masque_binaire(chemin):
    """
    Charge un masque Ground Truth binaire (noir/blanc).
    Retourne : masque (0 ou 1)
    """
    masque = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
    if masque is None:
        raise FileNotFoundError(f"Masque introuvable : {chemin}")
    _, masque_bin = cv2.threshold(masque, 127, 1, cv2.THRESH_BINARY)
    return masque_bin

def redimensionner_masque(masque, H, W):
    """
    Redimensionne un masque à la taille (H, W).
    Utilise INTER_NEAREST pour ne pas mélanger les couleurs.
    """
    return cv2.resize(masque, (W, H), interpolation=cv2.INTER_NEAREST)

# ══════════════════════════════════════════════
# 2. PRÉTRAITEMENT — TP4 (Filtres from scratch)
# ══════════════════════════════════════════════

def filtre_gaussien_noyau(taille=3, sigma=1.0):
    """
    Crée un noyau gaussien (TP4).
    G(x,y) = exp(-(x²+y²) / 2*sigma²)
    """
    k = taille // 2
    noyau = np.zeros((taille, taille))
    for x in range(-k, k+1):
        for y in range(-k, k+1):
            noyau[x+k, y+k] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return noyau / noyau.sum()

def convolution_canal(canal, noyau):
    """
    Applique une convolution 2D sur un canal (TP4).
    Ne pas utiliser cv2.filter2D — from scratch.
    """
    H, W = canal.shape
    N = noyau.shape[0]
    pad = N // 2
    img_pad = np.pad(canal, pad, mode='edge').astype(np.float64)
    res = np.zeros_like(canal, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            res[i, j] = np.sum(img_pad[i:i+N, j:j+N] * noyau)
    return np.clip(res, 0, 255).astype(np.uint8)

def appliquer_filtre_gaussien(img_rgb, taille=3, sigma=1.0):
    """
    Applique le filtre gaussien sur les 3 canaux RGB.
    Retourne l'image filtrée.
    """
    noyau = filtre_gaussien_noyau(taille, sigma)
    img_filtre = np.zeros_like(img_rgb)
    for c in range(3):
        img_filtre[:, :, c] = convolution_canal(img_rgb[:, :, c], noyau)
    return img_filtre

def filtre_median_canal(canal, taille=3):
    """
    Filtre médian from scratch sur un canal (TP4).
    Remplace chaque pixel par la médiane de ses voisins.
    """
    H, W = canal.shape
    pad = taille // 2
    img_pad = np.pad(canal, pad, mode='edge')
    res = np.zeros_like(canal)
    for i in range(H):
        for j in range(W):
            voisins = img_pad[i:i+taille, j:j+taille].flatten()
            res[i, j] = np.median(voisins)
    return res.astype(np.uint8)

# ══════════════════════════════════════════════
# 3. PRÉTRAITEMENT — TP2 (Histogramme)
# ══════════════════════════════════════════════

def calculer_histogramme(image_gray):
    """
    Calcule l'histogramme d'une image en niveaux de gris (TP1).
    Retourne un tableau de taille 256.
    """
    hist = np.zeros(256, dtype=int)
    for i in range(image_gray.shape[0]):
        for j in range(image_gray.shape[1]):
            hist[image_gray[i, j]] += 1
    return hist

def egalisation_histogramme(image_gray):
    """
    Égalisation d'histogramme from scratch (TP2).
    I' = round((CDF(I) - CDF_min) / (M*N - CDF_min) * (L-1))
    """
    M, N = image_gray.shape
    L = 256
    hist = calculer_histogramme(image_gray)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0][0]
    img_eq = np.zeros_like(image_gray)
    for i in range(M):
        for j in range(N):
            I = image_gray[i, j]
            img_eq[i, j] = round(
                (cdf[I] - cdf_min) / (M * N - cdf_min) * (L - 1)
            )
    return img_eq

# ══════════════════════════════════════════════
# 4. SEGMENTATION — TP3 (K-Means from scratch)
# ══════════════════════════════════════════════

def kmeans_from_scratch(pixels, K=4, max_iter=50, seed=42):
    """
    K-Means from scratch (TP3).
    Etapes : init -> assignation -> mise à jour -> convergence
    
    Paramètres :
        pixels   : array (N, D) — pixels normalisés
        K        : nombre de clusters
        max_iter : nombre max d'itérations
        seed     : graine aléatoire
    
    Retourne :
        labels  : array (N,) — cluster de chaque pixel
        centres : array (K, D) — centres finaux
    """
    np.random.seed(seed)

    # Étape a) Initialisation
    indices = np.random.choice(len(pixels), K, replace=False)
    centres = pixels[indices].copy()

    for iteration in range(max_iter):

        # Étape b) Assignation
        distances = np.sqrt(
            ((pixels[:, np.newaxis, :] - centres[np.newaxis, :, :]) ** 2
            ).sum(axis=2)
        )
        labels = np.argmin(distances, axis=1)

        # Étape c) Mise à jour
        nouveaux_centres = np.zeros_like(centres)
        for k in range(K):
            membres = pixels[labels == k]
            if len(membres) > 0:
                nouveaux_centres[k] = membres.mean(axis=0)
            else:
                nouveaux_centres[k] = centres[k]

        # Étape d) Convergence
        deplacement = np.sqrt(((nouveaux_centres - centres) ** 2).sum())
        centres = nouveaux_centres

        if deplacement < 1e-4:
            print(f"   K-Means convergé à l'itération {iteration+1}")
            break

    return labels, centres

# ══════════════════════════════════════════════
# 5. POST-TRAITEMENT — TP6 (Morphologie)
# ══════════════════════════════════════════════

def nettoyage_morphologique(masque_binaire, taille_kernel=5):
    """
    Nettoyage morphologique (TP6).
    1) Fermeture -> comble les trous
    2) Ouverture -> supprime le bruit
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (taille_kernel, taille_kernel)
    )
    masque = cv2.morphologyEx(masque_binaire, cv2.MORPH_CLOSE, kernel)
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, kernel)
    return masque

def otsu_from_scratch(image_gray):
    """
    Algorithme d'Otsu from scratch (TP6).
    Maximise la variance inter-classe : σ²b = w1 * w2 * (μ1 - μ2)²
    Retourne : masque_binaire, seuil_optimal
    """
    hist = calculer_histogramme(image_gray)
    total = image_gray.size
    hist_norm = hist / total

    meilleur_seuil = 0
    meilleure_variance = 0

    for T in range(1, 255):
        w1 = np.sum(hist_norm[:T+1])
        w2 = np.sum(hist_norm[T+1:])
        if w1 == 0 or w2 == 0:
            continue
        mu1 = np.sum(np.arange(T+1) * hist_norm[:T+1]) / w1
        mu2 = np.sum(np.arange(T+1, 256) * hist_norm[T+1:]) / w2
        variance = w1 * w2 * (mu1 - mu2) ** 2
        if variance > meilleure_variance:
            meilleure_variance = variance
            meilleur_seuil = T

    masque = (image_gray > meilleur_seuil).astype(np.uint8)
    print(f"   Seuil Otsu optimal : {meilleur_seuil}")
    return masque, meilleur_seuil

def composantes_connexes(masque_bin):
    """
    Étiquetage par composantes connexes (TP6).
    8-connectivité — BFS.
    Retourne : etiquettes, nombre_de_composantes
    """
    H, W = masque_bin.shape
    etiquettes = np.zeros((H, W), dtype=int)
    label_courant = 0
    voisins = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for i in range(H):
        for j in range(W):
            if masque_bin[i, j] > 0 and etiquettes[i, j] == 0:
                label_courant += 1
                file = [(i, j)]
                etiquettes[i, j] = label_courant
                while file:
                    ci, cj = file.pop(0)
                    for di, dj in voisins:
                        ni, nj = ci+di, cj+dj
                        if (0 <= ni < H and 0 <= nj < W and
                                masque_bin[ni, nj] > 0 and
                                etiquettes[ni, nj] == 0):
                            etiquettes[ni, nj] = label_courant
                            file.append((ni, nj))

    return etiquettes, label_courant

def garder_plus_grande_composante(masque_bin):
    """
    Garde uniquement la plus grande composante connexe.
    Utile pour isoler un objet principal.
    """
    etiquettes, nb = composantes_connexes(masque_bin)
    if nb == 0:
        return masque_bin
    tailles = [np.sum(etiquettes == i) for i in range(1, nb+1)]
    plus_grande = np.argmax(tailles) + 1
    return (etiquettes == plus_grande).astype(np.uint8)

# ══════════════════════════════════════════════
# 6. ÉVALUATION — Métriques
# ══════════════════════════════════════════════

def calculer_metriques(masque_gt, masque_pred, nom=""):
    """
    Calcule les métriques de segmentation.
    
    Métriques :
        - IoU (Intersection over Union)
        - Dice Coefficient
        - Précision
        - Rappel
        - F1-score
        - Pixel Accuracy
    
    Paramètres :
        masque_gt   : masque Ground Truth (0 ou 1)
        masque_pred : masque prédit (0 ou 1)
        nom         : nom de la classe/scène
    
    Retourne : dictionnaire des métriques
    """
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

def afficher_tableau_metriques(tous_resultats, titre="Résultats"):
    """
    Affiche un tableau récapitulatif des métriques dans le terminal.
    """
    print("\n" + "="*75)
    print(f"  {titre}")
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

    iou_moy  = np.mean([m['IoU']            for m in tous_resultats])
    dice_moy = np.mean([m['Dice']           for m in tous_resultats])
    f1_moy   = np.mean([m['F1-score']       for m in tous_resultats])
    acc_moy  = np.mean([m['Pixel Accuracy'] for m in tous_resultats])

    print("="*75)
    print(f"\n  {'MOYENNE':<12} "
          f"{iou_moy:>8.4f} "
          f"{dice_moy:>8.4f} "
          f"{'':>10} "
          f"{'':>8} "
          f"{f1_moy:>8.4f} "
          f"{acc_moy:>8.4f}")
    print("="*75)

    return iou_moy, dice_moy, f1_moy, acc_moy

# ══════════════════════════════════════════════
# 7. VISUALISATION
# ══════════════════════════════════════════════

def afficher_images(images, titres, figsize=(15, 5), sauvegarder=None):
    """
    Affiche plusieurs images côte à côte.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, titre in zip(axes, images, titres):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(titre)
        ax.axis('off')
    plt.tight_layout()
    if sauvegarder:
        plt.savefig(sauvegarder, dpi=150, bbox_inches='tight')
        print(f" Sauvegardé : {sauvegarder}")
    plt.show()

def sauvegarder_masque(masque, chemin):
    """
    Sauvegarde un masque binaire (0/1) en image PNG (0/255).
    """
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    cv2.imwrite(chemin, masque * 255)
    print(f" Masque sauvegardé : {chemin}")

def sauvegarder_image_rgb(img_rgb, chemin):
    """
    Sauvegarde une image RGB.
    """
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    cv2.imwrite(chemin, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print(f" Image sauvegardée : {chemin}")
