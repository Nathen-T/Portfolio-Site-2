from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image


def add_block_occlusion(X, block_width, block_height, img_height, img_width, random_state=None):
    X_occluded = X.copy()
    _, n_samples = X.shape
    rng = np.random.default_rng(random_state)
    for i in range(n_samples):
        img = X_occluded[:, i].reshape(img_height, img_width)
        max_x = img_width - block_width
        max_y = img_height - block_height
        if max_x > 0 and max_y > 0:
            x = rng.integers(0, max_x)
            y = rng.integers(0, max_y)
            img[y:y+block_height, x:x+block_width] = 255
            X_occluded[:, i] = img.ravel()
    return X_occluded


def add_noise(X, p, r, random_state=None):
    X_noisy = X.copy()
    rng = np.random.default_rng(random_state)
    n_features, n_samples = X.shape
    n_noisy_pixels = int(n_features * p)
    if n_noisy_pixels == 0:
        return X_noisy
    for i in range(n_samples):
        indices = rng.choice(n_features, n_noisy_pixels, replace=False)
        n_salt = int(n_noisy_pixels * r)
        salt_indices = rng.choice(indices, n_salt, replace=False)
        pepper_indices = np.setdiff1d(indices, salt_indices)
        X_noisy[salt_indices, i] = 255
        X_noisy[pepper_indices, i] = 0
    return X_noisy


def evaluate_clustering(H, Y_true):
    def assign_cluster_label(X, Y):
        kmeans = KMeans(n_clusters=len(set(Y)), random_state=42).fit(X)
        Y_pred = np.zeros(Y.shape)
        for i in set(kmeans.labels_):
            ind = kmeans.labels_ == i
            Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]
        return Y_pred
    Y_pred = assign_cluster_label(H.T, Y_true)
    acc = accuracy_score(Y_true, Y_pred)
    nmi = normalized_mutual_info_score(Y_true, Y_pred)
    return acc, nmi


def load_data(root='data/CroppedYaleB', reduce=4):
    images, labels = [], []
    for i, person in enumerate(sorted(os.listdir(root))):
        if not os.path.isdir(os.path.join(root, person)):
            continue
        for fname in os.listdir(os.path.join(root, person)):
            if fname.endswith('Ambient.pgm'):
                continue
            if not fname.endswith('.pgm'):
                continue
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L')
            img = img.resize([s//reduce for s in img.size])
            img = np.asarray(img).reshape((-1,1))
            images.append(img)
            labels.append(i)
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)
    return images, labels


def visualize_noise_addition(X, img_width, img_height):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(X[:, 0].reshape(img_height, img_width), cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    X_noisy_example = add_noise(X, p=0.1, r=0.5, random_state=42)
    ax[1].imshow(X_noisy_example[:, 0].reshape(img_height, img_width), cmap='gray')
    ax[1].set_title('Noisy Image (Salt & Pepper)')
    ax[1].axis('off')
    block_width = 12
    block_height = 12
    X_occluded_example = add_block_occlusion(X, block_width, block_height, img_height, img_width, random_state=42)
    ax[2].imshow(X_occluded_example[:, 0].reshape(img_height, img_width), cmap='gray')
    ax[2].set_title('Image with Block Occlusion')
    ax[2].axis('off')
    plt.show()

