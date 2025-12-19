"""
Script pour calculer l'influence sur un MLP avec EKFAC.

EKFAC (Eigenvalue-corrected KFAC) est une amélioration de KFAC qui corrige
les valeurs propres en utilisant les statistiques de gradient exactes.
"""

import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from datasets import load_dataset
import gensim.downloader as api
import torch
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

from deel.influenciae.common import InfluenceModel, EKFACIHVP
from deel.influenciae.influence import FirstOrderInfluenceCalculator
from deel.influenciae.utils import ORDER

MLP_PATH = "../models/notam_mlp_avgemb.pt"
DATASET_NAME = "DEEL-AI/NOTAM"
BATCH_SIZE = 32

# Parametres d'analyse d'influence
NUM_TRAIN_SAMPLES = None  # None = tous, ou un entier pour limiter
NUM_TEST_SAMPLES = 1     # Nombre d'exemples test a analyser
TOP_K = 5                 # Nombre d'exemples influents a afficher
BATCH_SIZE_HESSIAN = 4    # Petit batch pour le calcul de la Hessienne (evite OOM)
N_SAMPLES_HESSIAN = None  # Nombre d'exemples pour estimer la Hessienne (None = tous)

TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list:
    """Tokenisation simple."""
    return TOKEN_PATTERN.findall(text.lower())

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
    """
    Extrait les embeddings GloVe moyennes pour un dataset.
    """
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Extraction des embeddings GloVe pour {len(dataset)} exemples...")

    embeddings = []
    labels = []

    for i in tqdm(range(len(dataset))):
        example = dataset[i]
        emb = text_to_avg_embedding(example["text"], wv, emb_dim)
        embeddings.append(emb)
        labels.append(example["label"])

    embeddings = np.stack(embeddings)
    labels = np.array(labels)

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, labels

def create_tf_mlp(emb_dim: int, hidden_dim: int, num_classes: int) -> Model:
    """
    Cree un MLP TensorFlow equivalent au MLP PyTorch.
    Architecture: Linear(emb_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, num_classes)
    Note: On ignore le Dropout pour l'inference.
    """
    inputs = layers.Input(shape=(emb_dim,), name="embeddings")
    x = layers.Dense(hidden_dim, activation="relu", name="hidden")(inputs)
    outputs = layers.Dense(num_classes, activation=None, name="classifier")(x)

    model = Model(inputs=inputs, outputs=outputs, name="notam_mlp")
    return model


def transfer_mlp_weights(pt_checkpoint: dict, tf_model: Model) -> None:
    """
    Transfere les poids du MLP PyTorch vers le modele TensorFlow.
    """
    print("Transfert des poids PyTorch -> TensorFlow...")

    state_dict = pt_checkpoint["model_state_dict"]

    # Couche cachee (net.0)
    hidden_weight = state_dict["net.0.weight"].numpy()  # (hidden_dim, emb_dim)
    hidden_bias = state_dict["net.0.bias"].numpy()      # (hidden_dim,)
    # Keras attend (emb_dim, hidden_dim), donc on transpose
    tf_model.get_layer("hidden").set_weights([hidden_weight.T, hidden_bias])

    # Couche de sortie (net.3)
    output_weight = state_dict["net.3.weight"].numpy()  # (num_classes, hidden_dim)
    output_bias = state_dict["net.3.bias"].numpy()      # (num_classes,)
    tf_model.get_layer("classifier").set_weights([output_weight.T, output_bias])

    print("Poids transferes avec succes!")

def prepare_tf_dataset(
    embeddings: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32
) -> tf.data.Dataset:
    """Prepare un tf.data.Dataset batche pour influenciae."""
    dataset = tf.data.Dataset.from_tensor_slices((
        tf.constant(embeddings, dtype=tf.float32),
        tf.constant(labels, dtype=tf.int32)
    ))
    dataset = dataset.batch(batch_size)
    return dataset


def process_batch_for_loss(batch):
    """Fonction de pretraitement du batch pour influenciae."""
    inputs, labels = batch
    sample_weights = tf.ones_like(labels, dtype=tf.float32)
    return inputs, labels, sample_weights

def compute_influence_with_ekfac(
    tf_model: Model,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 32,
    batch_size_hessian: int = 4,
    n_samples_hessian: Optional[int] = 1000,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Calcule les scores d'influence en utilisant influenciae avec EKFAC.

    EKFAC utilise le pi-damping interne pour la regularisation,
    donc aucun parametre de damping n'est necessaire.
    """
    print(f"\n{'='*60}")
    print("Calcul d'influence avec influenciae (EKFAC)")
    print(f"{'='*60}")

    # Preparer les datasets TensorFlow
    train_dataset = prepare_tf_dataset(train_embeddings, train_labels, batch_size)
    test_dataset = prepare_tf_dataset(test_embeddings, test_labels, batch_size)

    # Dataset separe pour le calcul de la Hessienne (petit batch, subset)
    if n_samples_hessian is not None:
        hessian_embeddings = train_embeddings[:n_samples_hessian]
        hessian_labels = train_labels[:n_samples_hessian]
    else:
        hessian_embeddings = train_embeddings
        hessian_labels = train_labels
    hessian_dataset = prepare_tf_dataset(hessian_embeddings, hessian_labels, batch_size_hessian)

    # Loss function SANS reduction (requis par influenciae)
    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)

    # Wrapper le modele avec InfluenceModel
    # On surveille TOUS les poids du MLP (pas seulement la derniere couche)
    influence_model = InfluenceModel(
        tf_model,
        start_layer=1,   # Premiere couche Dense (hidden)
        last_layer=-1,   # Derniere couche Dense (classifier)
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch_for_loss
    )

    # Calculateur IHVP avec EKFAC
    # - update_eigen=True: utilise les eigenvalues corrigees (plus precis)
    # - Pas de parametre damping: EKFAC utilise le pi-damping interne
    print("Creation du calculateur IHVP (EKFAC)...")
    print(f"Nombre de parametres surveilles: {influence_model.nb_params}")
    print(f"Exemples pour Hessienne: {len(hessian_embeddings)}, batch_size: {batch_size_hessian}")

    ihvp_calculator = EKFACIHVP(
        model=influence_model,
        train_dataset=hessian_dataset,
        update_eigen=True  # Mise a jour des eigenvalues pour plus de precision
    )

    # Diagnostic: check EKFAC blocks properties
    for layer_idx, (A, G) in ihvp_calculator.kfac_blocks.items():
        layer_info = ihvp_calculator.layer_info[layer_idx]
        evals = ihvp_calculator.evals[layer_idx]
        print(f"Layer {layer_idx} ({layer_info['layer'].name}):")
        print(f"  A shape: {A.shape}, min: {tf.reduce_min(A).numpy():.6e}, max: {tf.reduce_max(A).numpy():.6e}")
        print(f"  G shape: {G.shape}, min: {tf.reduce_min(G).numpy():.6e}, max: {tf.reduce_max(G).numpy():.6e}")
        print(f"  Eigenvalues shape: {evals.shape}, min: {tf.reduce_min(evals).numpy():.6e}, max: {tf.reduce_max(evals).numpy():.6e}")

    print("Creation du FirstOrderInfluenceCalculator...")
    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_dataset,
        ihvp_calculator
    )

    # Diagnostic: Test computing influence vector for one batch
    print("\n=== Diagnostic: Influence vector computation ===")
    for batch in train_dataset.take(1):
        # Test batch_jacobian_tensor
        test_jacobian = influence_model.batch_jacobian_tensor(batch)
        print(f"Jacobian shape: {test_jacobian.shape}")
        print(f"Jacobian min: {tf.reduce_min(test_jacobian).numpy():.6e}")
        print(f"Jacobian max: {tf.reduce_max(test_jacobian).numpy():.6e}")
        print(f"Jacobian has NaN: {tf.reduce_any(tf.math.is_nan(test_jacobian)).numpy()}")
        print(f"Jacobian has Inf: {tf.reduce_any(tf.math.is_inf(test_jacobian)).numpy()}")

        # Test IHVP computation
        test_ihvp = ihvp_calculator._compute_ihvp_single_batch(batch)
        print(f"IHVP shape: {test_ihvp.shape}")
        print(f"IHVP min: {tf.reduce_min(test_ihvp).numpy():.6e}")
        print(f"IHVP max: {tf.reduce_max(test_ihvp).numpy():.6e}")
        print(f"IHVP has NaN: {tf.reduce_any(tf.math.is_nan(test_ihvp)).numpy()}")
        print(f"IHVP has Inf: {tf.reduce_any(tf.math.is_inf(test_ihvp)).numpy()}")

        # Test influence vector (which includes normalization)
        test_inf_vec = influence_calculator._compute_influence_vector(batch)
        print(f"Influence vector shape: {test_inf_vec.shape}")
        print(f"Influence vector min: {tf.reduce_min(test_inf_vec).numpy():.6e}")
        print(f"Influence vector max: {tf.reduce_max(test_inf_vec).numpy():.6e}")
        print(f"Influence vector has NaN: {tf.reduce_any(tf.math.is_nan(test_inf_vec)).numpy()}")
        print(f"Influence vector has Inf: {tf.reduce_any(tf.math.is_inf(test_inf_vec)).numpy()}")

    # Calculer le top-k des exemples influents (positifs) pour chaque exemple de test
    print(f"\nRecherche des top-{top_k} exemples influents (proponents) pour chaque exemple de test...")

    top_k_positive = influence_calculator.top_k(
        test_dataset,
        train_dataset,
        k=top_k,
        order=ORDER.DESCENDING
    )

    # Collecter les resultats positifs
    results = {
        "test_embeddings": [],
        "test_labels": [],
        "top_k_values": [],
        "top_k_train_samples": [],
        "bottom_k_values": [],
        "bottom_k_train_samples": []
    }

    for batch_result in top_k_positive:
        test_samples, influence_values, training_samples = batch_result
        test_emb, test_lab = test_samples

        results["test_embeddings"].append(test_emb.numpy())
        results["test_labels"].append(test_lab.numpy())
        results["top_k_values"].append(influence_values.numpy())

        if isinstance(training_samples, tuple):
            results["top_k_train_samples"].append(training_samples[0].numpy())
        else:
            results["top_k_train_samples"].append(training_samples.numpy())

    # Calculer le top-k des exemples avec influence negative (opponents)
    print(f"\nRecherche des top-{top_k} exemples avec influence negative (opponents)...")

    top_k_negative = influence_calculator.top_k(
        test_dataset,
        train_dataset,
        k=top_k,
        order=ORDER.ASCENDING
    )

    for batch_result in top_k_negative:
        test_samples, influence_values, training_samples = batch_result

        results["bottom_k_values"].append(influence_values.numpy())

        if isinstance(training_samples, tuple):
            results["bottom_k_train_samples"].append(training_samples[0].numpy())
        else:
            results["bottom_k_train_samples"].append(training_samples.numpy())

    return results


def find_train_indices(
    top_k_embeddings: np.ndarray,
    train_embeddings: np.ndarray
) -> np.ndarray:
    """
    Retrouve les indices des exemples train a partir de leurs embeddings.
    """
    # Calcul de distance euclidienne
    # top_k_embeddings: (top_k, emb_dim)
    # train_embeddings: (n_train, emb_dim)
    distances = np.linalg.norm(
        top_k_embeddings[:, np.newaxis, :] - train_embeddings[np.newaxis, :, :],
        axis=2
    )  # (top_k, n_train)
    indices = np.argmin(distances, axis=1)
    return indices


def display_influence_results(
    results: Dict[str, Any],
    train_dataset,
    test_dataset,
    train_embeddings: np.ndarray,
    label_names: list,
    top_k: int = 5
):
    """
    Affiche les resultats d'influence de maniere lisible.
    """
    print(f"\n{'='*80}")
    print("RESULTATS D'INFLUENCE (EKFAC)")
    print(f"{'='*80}")

    test_idx = 0

    for batch_idx, test_lab_batch in enumerate(results["test_labels"]):
        top_k_values = results["top_k_values"][batch_idx]
        top_k_train_emb = results["top_k_train_samples"][batch_idx]
        bottom_k_values = results["bottom_k_values"][batch_idx]
        bottom_k_train_emb = results["bottom_k_train_samples"][batch_idx]

        for i in range(len(test_lab_batch)):
            print(f"\n{'─'*80}")
            print(f"EXEMPLE DE TEST #{test_idx}")
            print(f"{'─'*80}")

            test_example = test_dataset[test_idx]
            test_label = test_lab_batch[i]
            print(f"Texte: {test_example['text'][:300]}...")
            print(f"Label: {test_label} ({label_names[test_label]})")

            # Afficher les proponents (influence positive)
            inf_vals_pos = top_k_values[i]
            train_embs_pos = top_k_train_emb[i]
            train_indices_pos = find_train_indices(train_embs_pos, train_embeddings)

            print(f"\n--- Top {top_k} exemples INFLUENTS (proponents) ---")

            for rank in range(min(top_k, len(inf_vals_pos))):
                score = inf_vals_pos[rank]
                train_idx = train_indices_pos[rank]
                train_example = train_dataset[int(train_idx)]

                print(f"\n  Rang #{rank + 1}")
                print(f"  Score d'influence: {score:.6f}")
                print(f"  Train idx: {train_idx}")
                print(f"  Label: {train_example['label']} ({label_names[train_example['label']]})")
                print(f"  Texte: {train_example['text']}...")

            # Afficher les opponents (influence negative)
            inf_vals_neg = bottom_k_values[i]
            train_embs_neg = bottom_k_train_emb[i]
            train_indices_neg = find_train_indices(train_embs_neg, train_embeddings)

            print(f"\n--- Top {top_k} exemples avec influence NEGATIVE (opponents) ---")

            for rank in range(min(top_k, len(inf_vals_neg))):
                score = inf_vals_neg[rank]
                train_idx = train_indices_neg[rank]
                train_example = train_dataset[int(train_idx)]

                print(f"\n  Rang #{rank + 1}")
                print(f"  Score d'influence: {score:.6f}")
                print(f"  Train idx: {train_idx}")
                print(f"  Label: {train_example['label']} ({label_names[train_example['label']]})")
                print(f"  Texte: {train_example['text']}...")

            test_idx += 1

            if test_idx >= 5:
                print("\n... (affichage limite aux 5 premiers exemples)")
                return


def main():
    print("="*60)
    print("ANALYSE D'INFLUENCE - MLP sur NOTAM (embeddings GloVe)")
    print("Methode: EKFAC (Eigenvalue-corrected KFAC)")
    print("="*60)

    # Charger le checkpoint PyTorch
    print(f"\nChargement du modele MLP: {MLP_PATH}")
    checkpoint = torch.load(MLP_PATH, map_location="cpu")

    emb_dim = checkpoint["emb_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    num_classes = checkpoint["num_classes"]
    label_names = checkpoint["label_names"]

    print(f"Architecture: {emb_dim} -> {hidden_dim} -> {num_classes}")
    print(f"Classes: {label_names}")

    # Charger les embeddings GloVe
    print("\nChargement des embeddings GloVe (100d) via gensim...")
    wv = api.load("glove-wiki-gigaword-100")
    print(f"Dimension des embeddings: {wv.vector_size}")

    # Charger le dataset
    print("\nChargement du dataset NOTAM...")
    dataset = load_dataset(DATASET_NAME)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print(f"Train set: {len(train_dataset)} exemples")
    print(f"Test set: {len(test_dataset)} exemples")

    # Extraire les embeddings GloVe
    train_embeddings, train_labels = extract_glove_embeddings(
        train_dataset,
        wv,
        emb_dim,
        num_samples=NUM_TRAIN_SAMPLES
    )

    test_embeddings, test_labels = extract_glove_embeddings(
        test_dataset,
        wv,
        emb_dim,
        num_samples=NUM_TEST_SAMPLES
    )

    # Creer le modele TensorFlow
    print("\nCreation du modele TensorFlow...")
    tf_model = create_tf_mlp(emb_dim, hidden_dim, num_classes)
    tf_model.summary()

    # Transferer les poids
    transfer_mlp_weights(checkpoint, tf_model)

    # Verifier que le modele fonctionne
    tf_model.compile(
        optimizer="adam",
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    # Test rapide sur les embeddings de test
    test_ds = prepare_tf_dataset(test_embeddings, test_labels, BATCH_SIZE)
    loss, acc = tf_model.evaluate(test_ds, verbose=0)
    print(f"Verification - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Calculer l'influence avec EKFAC
    results = compute_influence_with_ekfac(
        tf_model,
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        batch_size=BATCH_SIZE,
        batch_size_hessian=BATCH_SIZE_HESSIAN,
        n_samples_hessian=N_SAMPLES_HESSIAN,
        top_k=TOP_K
    )

    # Limiter les datasets pour l'affichage
    if NUM_TRAIN_SAMPLES is not None:
        train_subset = train_dataset.select(range(min(NUM_TRAIN_SAMPLES, len(train_dataset))))
    else:
        train_subset = train_dataset

    if NUM_TEST_SAMPLES is not None:
        test_subset = test_dataset.select(range(min(NUM_TEST_SAMPLES, len(test_dataset))))
    else:
        test_subset = test_dataset

    # Afficher les resultats
    display_influence_results(
        results,
        train_subset,
        test_subset,
        train_embeddings,
        label_names,
        top_k=TOP_K
    )

    print("\n" + "="*60)
    print("Analyse terminee!")
    print("="*60)


if __name__ == "__main__":
    main()
