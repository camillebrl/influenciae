"""
Script pour calculer l'influence par token sur un MLP avec embeddings moyennes (EKFAC).

Pour un MLP qui prend des embeddings moyennes en entree, le tokenwise influence
calcule comment chaque token du texte original contribue a l'influence totale.

La formule est:
    influence_token[i] = d/d(embedding_token_i) [ihvp^T @ grad_theta L(x_train, y_train)]

Ou l'embedding d'entree du MLP est: avg_emb = mean(embedding_token_i for i in tokens)
"""

import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from datasets import load_dataset
import gensim.downloader as api
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, List

from deel.influenciae.common import EKFACIHVP
from deel.influenciae.common.model_wrappers import BaseInfluenceModel
from deel.influenciae.influence import TokenwiseInfluenceCalculator

MLP_PATH = "../models/notam_mlp_avgemb.pt"
DATASET_NAME = "DEEL-AI/NOTAM"

# Parametres
BATCH_SIZE = 32
NUM_TRAIN_SAMPLES = None
NUM_TEST_SAMPLES = 1
TOP_K = 5
N_SAMPLES_HESSIAN = None

# GloVe
GLOVE_DIM = 100

TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    """Tokenisation simple."""
    return TOKEN_PATTERN.findall(text.lower())


class PyTorchMLP(nn.Module):
    """MLP PyTorch equivalent au modele sauvegarde."""
    def __init__(self, emb_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def load_pytorch_mlp(checkpoint_path: str, emb_dim: int, hidden_dim: int, num_classes: int) -> PyTorchMLP:
    """Charge le MLP PyTorch depuis un checkpoint."""
    model = PyTorchMLP(emb_dim, hidden_dim, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_token_embeddings(text: str, wv, dim: int) -> Tuple[List[str], np.ndarray]:
    """
    Retourne les tokens et leurs embeddings individuels.

    Returns
    -------
    tokens : list of str
        Liste des tokens trouves dans le vocabulaire GloVe
    embeddings : np.ndarray
        Embeddings de shape (n_tokens, dim)
    """
    raw_tokens = tokenize(text)
    tokens = []
    embeddings = []

    for t in raw_tokens:
        if t in wv:
            tokens.append(t)
            embeddings.append(wv[t])

    if not embeddings:
        return [], np.zeros((0, dim), dtype=np.float32)

    return tokens, np.stack(embeddings).astype(np.float32)


def text_to_avg_embedding(text: str, wv, dim: int) -> np.ndarray:
    """Convertit un texte en embedding moyen GloVe."""
    tokens = tokenize(text)
    vecs = []
    for t in tokens:
        if t in wv:
            vecs.append(wv[t])
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def extract_glove_embeddings(
    dataset,
    wv,
    emb_dim: int,
    num_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrait les embeddings GloVe moyennes pour un dataset."""
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    embeddings = []
    labels = []

    for i in tqdm(range(len(dataset)), desc="Extraction embeddings"):
        example = dataset[i]
        emb = text_to_avg_embedding(example["text"], wv, emb_dim)
        embeddings.append(emb)
        labels.append(example["label"])

    return np.stack(embeddings), np.array(labels)


def create_tf_mlp(emb_dim: int, hidden_dim: int, num_classes: int) -> Model:
    """Cree un MLP TensorFlow equivalent."""
    inputs = layers.Input(shape=(emb_dim,), name="embeddings")
    x = layers.Dense(hidden_dim, activation="relu", name="hidden")(inputs)
    outputs = layers.Dense(num_classes, activation=None, name="classifier")(x)
    return Model(inputs=inputs, outputs=outputs, name="notam_mlp")


def transfer_mlp_weights(pt_checkpoint: dict, tf_model: Model) -> None:
    """Transfere les poids du MLP PyTorch vers TensorFlow."""
    state_dict = pt_checkpoint["model_state_dict"]

    # Couche cachee (net.0)
    hidden_weight = state_dict["net.0.weight"].numpy()
    hidden_bias = state_dict["net.0.bias"].numpy()
    tf_model.get_layer("hidden").set_weights([hidden_weight.T, hidden_bias])

    # Couche de sortie (net.3)
    output_weight = state_dict["net.3.weight"].numpy()
    output_bias = state_dict["net.3.bias"].numpy()
    tf_model.get_layer("classifier").set_weights([output_weight.T, output_bias])


def prepare_tf_dataset(
    embeddings: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32
) -> tf.data.Dataset:
    """Prepare un tf.data.Dataset batche."""
    dataset = tf.data.Dataset.from_tensor_slices((
        tf.constant(embeddings, dtype=tf.float32),
        tf.constant(labels, dtype=tf.int32)
    ))
    return dataset.batch(batch_size)


def process_batch_for_loss(batch):
    """Pretraitement du batch pour influenciae."""
    inputs, labels = batch
    sample_weights = tf.ones_like(labels, dtype=tf.float32)
    return inputs, labels, sample_weights


def compute_tokenwise_influence_mlp(
    pt_model: PyTorchMLP,
    wv,
    emb_dim: int,
    x_train_text: str,
    y_train: int,
    ihvp_vec: np.ndarray
) -> Tuple[np.ndarray, List[str], float]:
    """
    Calcule l'influence par token pour un MLP avec embeddings moyennes.

    On backpropage a travers:
    1. Les parametres du MLP (pour obtenir grad_theta)
    2. L'operation de moyenne
    3. Jusqu'aux embeddings individuels de chaque token

    Returns
    -------
    influence_per_token : np.ndarray
        Influence par token (norme L2 du gradient)
    tokens : list of str
        Liste des tokens
    total_influence : float
        Influence totale
    """
    pt_model.eval()

    # Obtenir les embeddings individuels pour chaque token
    tokens, token_embeddings = get_token_embeddings(x_train_text, wv, emb_dim)

    if len(tokens) == 0:
        return np.array([]), [], 0.0

    # Convertir en tenseurs PyTorch avec gradients
    token_emb_tensor = torch.tensor(token_embeddings, requires_grad=True)  # (n_tokens, emb_dim)

    # Moyenne des embeddings (comme dans l'entrainement)
    avg_embedding = token_emb_tensor.mean(dim=0, keepdim=True)  # (1, emb_dim)

    # Forward pass
    logits = pt_model(avg_embedding)

    # Loss
    loss_fn = nn.CrossEntropyLoss()
    y_tensor = torch.tensor([y_train])
    loss = loss_fn(logits, y_tensor)

    # Gradient par rapport aux parametres du MLP
    grad_params = torch.autograd.grad(
        loss,
        pt_model.parameters(),
        create_graph=True,
        retain_graph=True
    )

    # Flatten les gradients
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_params])

    # IHVP en tensor PyTorch
    ihvp_tensor = torch.tensor(ihvp_vec, dtype=torch.float32)

    # Produit scalaire: ihvp^T @ grad_theta
    influence_scalar = torch.dot(ihvp_tensor, grad_vec)

    # Backprop vers les embeddings de tokens individuels
    influence_grad = torch.autograd.grad(
        influence_scalar,
        token_emb_tensor,
        retain_graph=False
    )[0]  # Shape: (n_tokens, emb_dim)

    # Norme L2 par token
    influence_per_token = torch.norm(influence_grad, dim=1).detach().numpy()

    total_influence = influence_scalar.item()

    return influence_per_token, tokens, total_influence


def main():
    print("="*60)
    print("ANALYSE D'INFLUENCE PAR TOKEN - MLP (EKFAC)")
    print("="*60)

    # Charger GloVe
    print("\nChargement de GloVe...")
    wv = api.load("glove-wiki-gigaword-100")
    emb_dim = wv.vector_size
    print(f"GloVe dim: {emb_dim}")

    # Charger le dataset
    print("\nChargement du dataset...")
    dataset = load_dataset(DATASET_NAME)
    train_ds_hf = dataset["train"]
    test_ds_hf = dataset["test"]

    # Labels
    label_names = list(set(train_ds_hf["label"]))
    label_names.sort()
    label2id = {name: i for i, name in enumerate(label_names)}
    num_classes = len(label_names)

    print(f"Classes ({num_classes}): {label_names}")

    # Limiter les exemples
    if NUM_TRAIN_SAMPLES is not None:
        train_ds_hf = train_ds_hf.select(range(min(NUM_TRAIN_SAMPLES, len(train_ds_hf))))
    if NUM_TEST_SAMPLES is not None:
        test_ds_hf = test_ds_hf.select(range(min(NUM_TEST_SAMPLES, len(test_ds_hf))))

    print(f"Train: {len(train_ds_hf)} exemples, Test: {len(test_ds_hf)} exemples")

    # Charger le MLP PyTorch
    print(f"\nChargement du modele PyTorch: {MLP_PATH}")
    pt_checkpoint = torch.load(MLP_PATH, map_location="cpu", weights_only=False)

    # Determiner l'architecture depuis le checkpoint
    state_dict = pt_checkpoint["model_state_dict"]
    hidden_dim = state_dict["net.0.weight"].shape[0]
    print(f"Architecture: {emb_dim} -> {hidden_dim} -> {num_classes}")

    pt_model = load_pytorch_mlp(MLP_PATH, emb_dim, hidden_dim, num_classes)

    # Convertir les labels
    train_labels = np.array([label2id[lbl] for lbl in train_ds_hf["label"]], dtype=np.int32)
    test_labels = np.array([label2id[lbl] for lbl in test_ds_hf["label"]], dtype=np.int32)

    train_texts = list(train_ds_hf["text"])
    test_texts = list(test_ds_hf["text"])

    # Extraire les embeddings moyennes
    print("\nExtraction des embeddings...")
    train_embeddings, _ = extract_glove_embeddings(train_ds_hf, wv, emb_dim, NUM_TRAIN_SAMPLES)
    test_embeddings, _ = extract_glove_embeddings(test_ds_hf, wv, emb_dim, NUM_TEST_SAMPLES)

    print(f"Train embeddings: {train_embeddings.shape}")
    print(f"Test embeddings: {test_embeddings.shape}")

    # Creer le modele TF
    print("\nCreation du modele TensorFlow...")
    tf_model = create_tf_mlp(emb_dim, hidden_dim, num_classes)
    transfer_mlp_weights(pt_checkpoint, tf_model)

    # Preparer le dataset pour EKFAC
    hessian_embeddings = train_embeddings[:N_SAMPLES_HESSIAN]
    hessian_labels = train_labels[:N_SAMPLES_HESSIAN]
    hessian_dataset = prepare_tf_dataset(hessian_embeddings, hessian_labels, batch_size=1)

    # Loss function
    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)

    # Build le modele
    for batch in hessian_dataset.take(1):
        inputs, _ = batch
        _ = tf_model(inputs, training=True)

    # Collecter les poids Dense
    dense_weights = []
    for layer in tf_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            dense_weights.extend(layer.trainable_weights)

    # Wrapper le modele
    influence_model = BaseInfluenceModel(
        tf_model,
        weights_to_watch=dense_weights,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch_for_loss,
        weights_processed=True
    )

    print(f"Nombre de parametres surveilles: {influence_model.nb_params}")

    # Creer le calculateur EKFAC
    print("\nCreation du calculateur EKFAC...")
    ekfac_calculator = EKFACIHVP(
        model=influence_model,
        train_dataset=hessian_dataset,
        update_eigen=True
    )

    # Afficher les diagnostics EKFAC
    print("\n=== EKFAC blocks ===")
    for layer_idx, (A, G) in ekfac_calculator.kfac_blocks.items():
        layer_info = ekfac_calculator.layer_info[layer_idx]
        evals = ekfac_calculator.evals[layer_idx]
        print(f"Layer {layer_idx} ({layer_info['layer'].name}):")
        print(f"  A: {A.shape}, G: {G.shape}")
        print(f"  Eigenvalues: min={tf.reduce_min(evals).numpy():.2e}, max={tf.reduce_max(evals).numpy():.2e}")

    # === Test du TokenwiseInfluenceCalculator TensorFlow ===
    print("\n" + "="*60)
    print("TEST: TokenwiseInfluenceCalculator (TensorFlow)")
    print("="*60)

    tokenwise_calc = TokenwiseInfluenceCalculator(
        model=influence_model,
        ihvp_calculator=ekfac_calculator
    )

    # Tester sur le premier exemple
    x_test_tf = tf.constant(test_embeddings[0:1], dtype=tf.float32)
    y_test_tf = tf.constant(test_labels[0:1], dtype=tf.int32)
    x_train_tf = tf.constant(train_embeddings[0:1], dtype=tf.float32)
    y_train_tf = tf.constant(train_labels[0:1], dtype=tf.int32)

    # Influence totale via TokenwiseInfluenceCalculator
    total_inf_tf = tokenwise_calc.compute_total_influence(
        x_train_tf, y_train_tf, x_test_tf, y_test_tf
    )
    print(f"\nInfluence totale (TF TokenwiseCalculator): {total_inf_tf.numpy():.6f}")

    # Influence map (sur l'embedding moyenne - pas par token)
    inf_map_tf = tokenwise_calc.compute_tokenwise_influence(
        x_train_tf, y_train_tf, x_test_tf, y_test_tf
    )
    print(f"Influence map shape: {inf_map_tf.shape}")
    print(f"Influence map norm: {tf.norm(inf_map_tf).numpy():.6f}")

    # === Analyse par token avec PyTorch ===
    print("\n" + "="*60)
    print("ANALYSE PAR TOKEN (via PyTorch)")
    print("="*60)

    for test_idx in range(len(test_ds_hf)):
        print(f"\n{'='*80}")
        print(f"EXEMPLE DE TEST #{test_idx}")
        print(f"{'='*80}")

        test_example = test_ds_hf[test_idx]
        test_text = test_example["text"]
        test_label = test_labels[test_idx]

        print(f"Texte: {test_text}...")
        print(f"Label: {test_label} ({label_names[test_label]})")

        # Calculer l'IHVP pour l'exemple de test
        x_test = tf.constant(test_embeddings[test_idx:test_idx+1], dtype=tf.float32)
        y_test = tf.constant(test_labels[test_idx:test_idx+1], dtype=tf.int32)
        test_batch = (x_test, y_test)

        ihvp = ekfac_calculator._compute_ihvp_single_batch(test_batch)
        ihvp_vec = tf.reshape(ihvp, [-1]).numpy()

        print(f"\nIHVP shape: {ihvp_vec.shape}")
        print(f"IHVP norm: {np.linalg.norm(ihvp_vec):.6e}")

        # Calculer l'influence totale pour chaque exemple d'entrainement
        print("\nCalcul des influences totales...")
        influences = []

        for train_idx in tqdm(range(len(train_ds_hf))):
            x_train = tf.constant(train_embeddings[train_idx:train_idx+1], dtype=tf.float32)
            y_train = tf.constant(train_labels[train_idx:train_idx+1], dtype=tf.int32)

            with tf.GradientTape() as tape:
                predictions = tf_model(x_train, training=True)
                loss = loss_fn(y_train, predictions)
                loss = tf.reduce_sum(loss)

            grad_train = tape.gradient(loss, tf_model.trainable_variables)
            grad_vec = tf.concat([tf.reshape(g, [-1]) for g in grad_train], axis=0).numpy()

            influence = np.dot(ihvp_vec, grad_vec)
            influences.append(influence)

        influences = np.array(influences)

        # Top-k proponents et opponents
        top_k_idx = np.argsort(influences)[-TOP_K:][::-1]
        bottom_k_idx = np.argsort(influences)[:TOP_K]

        print(f"\n--- Top {TOP_K} PROPONENTS ---")
        for rank, train_idx in enumerate(top_k_idx):
            train_example = train_ds_hf[int(train_idx)]
            print(f"\n  Rang #{rank + 1}")
            print(f"  Score: {influences[train_idx]:.6f}")
            print(f"  Label: {train_example['label']}")
            print(f"  Texte: {train_example['text']}...")

            # Calculer l'influence par token
            try:
                influence_per_token, tokens, total_inf = compute_tokenwise_influence_mlp(
                    pt_model, wv, emb_dim,
                    train_example["text"],
                    label2id[train_example["label"]],
                    ihvp_vec
                )

                if len(tokens) > 0:
                    # Top 5 tokens les plus influents
                    top_token_idx = np.argsort(influence_per_token)[-5:][::-1]
                    print(f"  Tokens les plus influents:")
                    for ti in top_token_idx:
                        print(f"    - '{tokens[ti]}': {influence_per_token[ti]:.4f}")
            except Exception as e:
                print(f"  Erreur tokenwise: {e}")

        print(f"\n--- Top {TOP_K} OPPONENTS ---")
        for rank, train_idx in enumerate(bottom_k_idx):
            train_example = train_ds_hf[int(train_idx)]
            print(f"\n  Rang #{rank + 1}")
            print(f"  Score: {influences[train_idx]:.6f}")
            print(f"  Label: {train_example['label']}")
            print(f"  Texte: {train_example['text']}...")

            # Calculer l'influence par token
            try:
                influence_per_token, tokens, total_inf = compute_tokenwise_influence_mlp(
                    pt_model, wv, emb_dim,
                    train_example["text"],
                    label2id[train_example["label"]],
                    ihvp_vec
                )

                if len(tokens) > 0:
                    top_token_idx = np.argsort(influence_per_token)[-5:][::-1]
                    print(f"  Tokens les plus influents:")
                    for ti in top_token_idx:
                        print(f"    - '{tokens[ti]}': {influence_per_token[ti]:.4f}")
            except Exception as e:
                print(f"  Erreur tokenwise: {e}")

    print("\n" + "="*60)
    print("Analyse terminee!")
    print("="*60)


if __name__ == "__main__":
    main()
