"""
Script pour calculer l'influence sur un transformer finetuné.

Approche:
1. Charger le transformer PyTorch (ModernBERT)
2. Extraire les embeddings ou représentations intermédiaires
3. Créer un MLP TensorFlow équivalent (classifier seul ou avec MLP encoder)
4. Transférer les poids PyTorch vers TensorFlow
5. Calculer l'influence avec KFAC sur le MLP TensorFlow
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from datasets import load_dataset
from typing import Optional, Dict, Any, List
import torch
from tqdm import tqdm

import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from deel.influenciae.common import KFACIHVP
from deel.influenciae.common.model_wrappers import BaseInfluenceModel
from deel.influenciae.influence import FirstOrderInfluenceCalculator
from deel.influenciae.utils import ORDER


DATASET_NAME = "DEEL-AI/NOTAM"
MODEL_NAME = "../models/modernbert_finetuned_notam"

MAX_LEN = 256
BATCH_SIZE = 1  # Réduit pour économiser la mémoire GPU
BATCH_SIZE_HESSIAN = 1  # Réduit à 1 pour le calcul KFAC sur grand modèle

# Influence
NUM_TRAIN_SAMPLES = None
NUM_TEST_SAMPLES = 1
TOP_K = 5
N_SAMPLES_HESSIAN = None

# Mode: "classifier_only" ou "with_encoder_mlp"
# - classifier_only: influence sur le classifier head uniquement
# - with_encoder_mlp: influence sur le(s) MLP de l'encodeur + classifier
INFLUENCE_MODE = "classifier_only"

# Layers à inclure pour with_encoder_mlp (None = tous, ou liste comme [20, 21])
ENCODER_LAYERS = [21]  # 1 seul layer pour éviter OOM 
# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_transformer_model(model_name: str):
    """Charge le modèle transformer PyTorch et le tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    return model, tokenizer


def debug_print_model_modules(pt_model, filter_str: str = None):
    """Affiche tous les modules du modèle pour debug."""
    print("\n=== Debug: Modules du modèle PyTorch ===")
    for name, module in pt_model.named_modules():
        if filter_str is None or filter_str in name:
            print(f"  {name}: {type(module).__name__}")
    print("=" * 50)


def get_encoder_mlp_info(pt_model) -> List[Dict[str, Any]]:
    """
    Extrait les informations sur les couches MLP de l'encodeur.

    Pour ModernBERT et BERT-like models, les MLPs sont dans:
    - model.encoder.layer[i].intermediate.dense (expansion)
    - model.encoder.layer[i].output.dense (projection)
    Ou pour ModernBERT:
    - model.layers[i].mlp.Wi, model.layers[i].mlp.Wo
    """
    
    _MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(Wi|Wo)$")

    mlp_layers: List[Dict[str, Any]] = []

    for name, module in pt_model.named_modules():
        m = _MLP_RE.match(name)
        if m is None:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue

        layer_idx = int(m.group(1))
        which = m.group(2)  # "Wi" or "Wo"

        mlp_layers.append({
            "layer_idx": layer_idx,
            "which": which,
            "name": name,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "weight": module.weight.detach().cpu().numpy(),
            "bias": None if module.bias is None else module.bias.detach().cpu().numpy(),
        })

    # tri stable: couche 0..21, puis Wi avant Wo
    mlp_layers.sort(key=lambda d: (d["layer_idx"], 0 if d["which"] == "Wi" else 1))
    return mlp_layers


def get_classifier_info(pt_model) -> Dict[str, Any]:
    """
    Extrait les informations sur le classifier head du modèle.
    """
    classifier = None
    classifier_name = None

    for name in ['classifier', 'score', 'cls', 'head']:
        if hasattr(pt_model, name):
            classifier = getattr(pt_model, name)
            classifier_name = name
            break

    if classifier is None:
        raise ValueError("Impossible de trouver le classifier dans le modèle")

    print(f"Classifier trouvé: {classifier_name}")
    print(f"Structure: {classifier}")

    # Extraire les poids selon le type
    if isinstance(classifier, torch.nn.Linear):
        weight = classifier.weight.detach().cpu().numpy()
        bias = classifier.bias.detach().cpu().numpy() if classifier.bias is not None else None

        return {
            "type": "linear",
            "in_features": classifier.in_features,
            "out_features": classifier.out_features,
            "weights": [(weight, bias)],
            "layers": [{"name": "classifier", "in_features": classifier.in_features, "out_features": classifier.out_features}]
        }
    else:
        # Sequential ou custom - extraire toutes les couches Linear
        weights = []
        layers_info = []

        for name, module in classifier.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.detach().cpu().numpy()
                bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
                weights.append((weight, bias))
                layers_info.append({
                    "name": name if name else "linear",
                    "in_features": module.in_features,
                    "out_features": module.out_features
                })

        if not weights:
            raise ValueError(f"Pas de couches Linear trouvées dans le classifier: {type(classifier)}")

        return {
            "type": "sequential",
            "layers": layers_info,
            "weights": weights
        }


def extract_embeddings_from_transformer(
    pt_model,
    tokenizer,
    texts: list,
    max_len: int,
    batch_size: int = 32,
    layer_index: int = -1  # -1 = dernière couche avant classifier
) -> np.ndarray:
    """
    Extrait les embeddings du transformer.
    """
    pt_model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extraction embeddings"):
        batch_texts = texts[i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            # Obtenir les outputs du modèle de base
            outputs = pt_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            # Récupérer le pooled output ou last_hidden_state[:, 0, :]
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def create_tf_model(classifier_info: Dict[str, Any], include_hidden: bool = False) -> Model:
    """
    Crée un modèle TensorFlow équivalent.

    Si include_hidden=True et il y a plusieurs couches, on les inclut toutes.
    Sinon, on crée juste le classifier.
    """
    layers_info = classifier_info.get("layers", [])

    if not layers_info:
        raise ValueError("Pas d'information sur les couches")

    # Déterminer la dimension d'entrée
    in_features = layers_info[0]["in_features"]

    inputs = layers.Input(shape=(in_features,), name="embeddings")
    x = inputs

    # Créer les couches
    for i, layer_info in enumerate(layers_info):
        is_last = (i == len(layers_info) - 1)

        # Activation ReLU sauf pour la dernière couche
        activation = None if is_last else "gelu"

        # Nom de la couche
        if is_last:
            name = "classifier"
        elif i == 0 and len(layers_info) > 1:
            name = "hidden"
        else:
            name = f"dense_{i}"

        x = layers.Dense(
            layer_info["out_features"],
            activation=activation,
            name=name
        )(x)

    model = Model(inputs=inputs, outputs=x, name="influence_model")
    return model


def transfer_weights_to_tf(classifier_info: Dict[str, Any], tf_model: Model) -> None:
    """
    Transfère les poids du classifier PyTorch vers TensorFlow.
    """
    print("Transfert des poids PyTorch -> TensorFlow...")

    weights_list = classifier_info["weights"]
    dense_layers = [layer for layer in tf_model.layers if isinstance(layer, tf.keras.layers.Dense)]

    if len(weights_list) != len(dense_layers):
        print(f"Warning: {len(weights_list)} poids PyTorch vs {len(dense_layers)} couches TF")

    for i, (weight, bias) in enumerate(weights_list):
        if i < len(dense_layers):
            layer = dense_layers[i]
            # PyTorch: (out_features, in_features), Keras: (in_features, out_features)
            if bias is not None:
                layer.set_weights([weight.T, bias])
            else:
                layer.set_weights([weight.T, np.zeros(weight.shape[0])])
            print(f"  {layer.name}: ({weight.shape[1]}, {weight.shape[0]})")

    print("Poids transférés avec succès!")


def prepare_tf_dataset(
    embeddings: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32
) -> tf.data.Dataset:
    """Prépare un tf.data.Dataset batché pour influenciae."""
    dataset = tf.data.Dataset.from_tensor_slices((
        tf.constant(embeddings, dtype=tf.float32),
        tf.constant(labels, dtype=tf.int32)
    ))
    dataset = dataset.batch(batch_size)
    return dataset


def process_batch_for_loss(batch):
    """Fonction de prétraitement du batch pour influenciae."""
    inputs, labels = batch
    sample_weights = tf.ones_like(labels, dtype=tf.float32)
    return inputs, labels, sample_weights


def compute_influence_with_kfac(
    tf_model: Model,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 32,
    batch_size_hessian: int = 4,
    n_samples_hessian: Optional[int] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Calcule les scores d'influence avec KFAC.
    """
    print(f"\n{'='*60}")
    print("Calcul d'influence avec KFAC")
    print(f"{'='*60}")

    # Préparer les datasets TensorFlow
    train_dataset = prepare_tf_dataset(train_embeddings, train_labels, batch_size)
    test_dataset = prepare_tf_dataset(test_embeddings, test_labels, batch_size)

    # Dataset pour le calcul de la Hessienne
    if n_samples_hessian is not None:
        hessian_embeddings = train_embeddings[:n_samples_hessian]
        hessian_labels = train_labels[:n_samples_hessian]
    else:
        hessian_embeddings = train_embeddings
        hessian_labels = train_labels
    hessian_dataset = prepare_tf_dataset(hessian_embeddings, hessian_labels, batch_size_hessian)

    # Loss function SANS réduction
    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)

    # IMPORTANT: Build le modèle avec un forward pass
    print("\nBuild du modèle TensorFlow...")
    for batch in train_dataset.take(1):
        inputs, _ = batch
        _ = tf_model(inputs, training=True)

    # Afficher la structure complète du modèle
    print("\nStructure complète du modèle:")
    for i, layer in enumerate(tf_model.layers):
        trainable = "trainable" if layer.trainable else "frozen"
        n_params = sum(tf.reduce_prod(w.shape).numpy() for w in layer.trainable_weights)
        layer_type = type(layer).__name__
        if hasattr(layer, 'units'):
            print(f"  [{i}] {layer.name} ({layer_type}): units={layer.units}, params={n_params} ({trainable})")
        else:
            print(f"  [{i}] {layer.name} ({layer_type}): params={n_params} ({trainable})")

    # Trouver les indices des couches Dense
    dense_indices = [i for i, layer in enumerate(tf_model.layers)
                     if isinstance(layer, tf.keras.layers.Dense)]

    if not dense_indices:
        raise ValueError("Pas de couches Dense trouvées dans le modèle")

    print(f"\nCouches Dense aux indices: {dense_indices}")

    # IMPORTANT: Geler toutes les couches non-Dense pour éviter les erreurs KFAC
    print("\nGel des couches non-Dense...")
    for i, layer in enumerate(tf_model.layers):
        if not isinstance(layer, tf.keras.layers.Dense):
            layer.trainable = False
            print(f"  Gelé: [{i}] {layer.name}")

    # Configurer start_layer et last_layer pour inclure toutes les Dense
    start_layer = dense_indices[0]
    last_layer = dense_indices[-1]

    print(f"\nSurveillance des couches [{start_layer}] à [{last_layer}]")

    # Collecter UNIQUEMENT les poids trainable des couches Dense
    # (évite d'inclure les poids des LayerNorm même si elles sont "gelées")
    dense_weights = []
    for layer in tf_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            dense_weights.extend(layer.trainable_weights)

    print(f"Poids Dense collectés: {len(dense_weights)} tenseurs")
    for w in dense_weights:
        print(f"  - {w.name}: {w.shape}")

    total_params = sum(tf.reduce_prod(w.shape).numpy() for w in dense_weights)
    print(f"Total paramètres Dense: {total_params}")

    # Wrapper le modèle avec BaseInfluenceModel et poids explicites
    influence_model = BaseInfluenceModel(
        tf_model,
        weights_to_watch=dense_weights,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch_for_loss,
        weights_processed=True  # Les poids sont déjà en liste plate
    )

    print(f"Nombre de paramètres surveillés: {influence_model.nb_params}")
    print(f"Exemples pour Hessienne: {len(hessian_embeddings)}, batch_size: {batch_size_hessian}")

    # Diagnostic: vérifier que le modèle produit des gradients
    print("\n=== Diagnostic: Vérification des gradients ===")
    for batch in hessian_dataset.take(1):
        inputs, labels, _ = process_batch_for_loss(batch)
        with tf.GradientTape() as tape:
            predictions = tf_model(inputs, training=True)
            loss = loss_fn(labels, predictions)

        grads = tape.gradient(loss, tf_model.trainable_variables)
        for var, grad in zip(tf_model.trainable_variables, grads):
            if grad is not None:
                print(f"  {var.name}: grad norm = {tf.norm(grad).numpy():.6e}")
            else:
                print(f"  {var.name}: grad = None")
    print("=" * 50)

    # Calculateur IHVP avec KFAC
    print("\nCréation du calculateur IHVP (KFAC)...")
    ihvp_calculator = KFACIHVP(
        model=influence_model,
        train_dataset=hessian_dataset
    )

    # Diagnostic: afficher les blocs KFAC
    print(f"\n=== Diagnostic: KFAC blocks ===")
    for layer_idx, (A, G) in ihvp_calculator.kfac_blocks.items():
        layer_info = ihvp_calculator.layer_info[layer_idx]
        print(f"Layer {layer_idx} ({layer_info['layer'].name}):")
        print(f"  A shape: {A.shape}, norm: {tf.norm(A).numpy():.6e}")
        print(f"  G shape: {G.shape}, norm: {tf.norm(G).numpy():.6e}")
    print(f"===============================\n")

    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_dataset,
        ihvp_calculator
    )

    # Calculer top-k proponents
    print(f"\nRecherche des top-{top_k} proponents...")
    top_k_positive = influence_calculator.top_k(
        test_dataset, train_dataset, k=top_k, order=ORDER.DESCENDING
    )
    print("top k : ", top_k_positive)
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

    # Calculer top-k opponents (influence négative)
    print(f"\nRecherche des top-{top_k} opponents...")
    top_k_negative = influence_calculator.top_k(
        test_dataset, train_dataset, k=top_k, order=ORDER.ASCENDING
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
    """Retrouve les indices des exemples train à partir de leurs embeddings."""
    distances = np.linalg.norm(
        top_k_embeddings[:, np.newaxis, :] - train_embeddings[np.newaxis, :, :],
        axis=2
    )
    indices = np.argmin(distances, axis=1)
    return indices


def display_results(
    results: Dict[str, Any],
    train_dataset,
    test_dataset,
    train_embeddings: np.ndarray,
    label_names: list,
    top_k: int = 5
):
    """Affiche les résultats d'influence."""
    print(f"\n{'='*80}")
    print("RÉSULTATS D'INFLUENCE (KFAC - Transformer)")
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

            # Proponents
            inf_vals_pos = top_k_values[i]
            train_embs_pos = top_k_train_emb[i]
            train_indices_pos = find_train_indices(train_embs_pos, train_embeddings)

            print(f"\n--- Top {top_k} PROPONENTS (influence positive) ---")
            for rank in range(min(top_k, len(inf_vals_pos))):
                score = inf_vals_pos[rank]
                train_idx = train_indices_pos[rank]
                train_example = train_dataset[int(train_idx)]

                print(f"\n  Rang #{rank + 1}")
                print(f"  Score: {score:.6f}")
                print(f"  Train idx: {train_idx}")
                print(f"  Label: {train_example['label']} ({label_names[train_example['label']]})")
                print(f"  Texte: {train_example['text']}...")

            # Opponents
            inf_vals_neg = bottom_k_values[i]
            train_embs_neg = bottom_k_train_emb[i]
            train_indices_neg = find_train_indices(train_embs_neg, train_embeddings)

            print(f"\n--- Top {top_k} OPPONENTS (influence négative) ---")
            for rank in range(min(top_k, len(inf_vals_neg))):
                score = inf_vals_neg[rank]
                train_idx = train_indices_neg[rank]
                train_example = train_dataset[int(train_idx)]

                print(f"\n  Rang #{rank + 1}")
                print(f"  Score: {score:.6f}")
                print(f"  Train idx: {train_idx}")
                print(f"  Label: {train_example['label']} ({label_names[train_example['label']]})")
                print(f"  Texte: {train_example['text']}...")

            test_idx += 1

            if test_idx >= 5:
                print("\n... (affichage limité aux 5 premiers exemples)")
                return


def extract_classifier_for_layer(pt_model, layer_idx: int) -> Dict[str, Any]:
    """
    Extrait uniquement le classifier pour calculer l'influence sur un layer spécifique.

    Puisque output_hidden_states retourne des embeddings de dimension hidden_size (768),
    on ne peut appliquer que le classifier (768 -> num_classes).

    L'influence calculée représente comment les exemples d'entraînement affectent
    la prédiction quand on utilise les représentations du layer spécifié.

    Parameters
    ----------
    pt_model : PreTrainedModel
        Le modèle PyTorch
    layer_idx : int
        Index du layer transformer (0 à 21 pour ModernBERT)
    """
    # Récupérer le classifier
    classifier_info = get_classifier_info(pt_model)

    # Obtenir la dimension d'entrée attendue
    hidden_size = classifier_info["layers"][0]["in_features"]

    print(f"  Layer {layer_idx} -> Classifier ({hidden_size} -> {classifier_info['layers'][0]['out_features']})")

    return {
        "type": "layer_classifier",
        "layers": classifier_info["layers"],
        "weights": classifier_info["weights"],
        "hidden_size": hidden_size,
        "layer_idx": layer_idx
    }


def extract_mlp_and_classifier_for_layer(pt_model, layer_idx: int) -> Dict[str, Any]:
    """
    Extrait Wi, Wo (MLP) et le classifier pour un layer spécifique.

    Permet de calculer l'influence sur toutes les couches Dense du MLP + classifier.

    Architecture du MLP ModernBERT (SwiGLU):
    - Wi: 768 -> 2304 (gate + up projection concaténés)
    - Gating: split en 2x1152, SiLU(gate) * up -> 1152
    - Wo: 1152 -> 768

    Parameters
    ----------
    pt_model : PreTrainedModel
        Le modèle PyTorch
    layer_idx : int
        Index du layer transformer (0 à 21 pour ModernBERT)
    """
    # Récupérer les couches MLP
    mlp_layers = get_encoder_mlp_info(pt_model)

    # Filtrer pour le layer demandé
    layer_mlp = [layer for layer in mlp_layers if layer["layer_idx"] == layer_idx]

    if len(layer_mlp) != 2:
        raise ValueError(f"Attendu 2 couches MLP (Wi, Wo) pour layer {layer_idx}, trouvé {len(layer_mlp)}")

    wi_info = next(layer for layer in layer_mlp if layer["which"] == "Wi")
    wo_info = next(layer for layer in layer_mlp if layer["which"] == "Wo")

    # Récupérer le classifier
    classifier_info = get_classifier_info(pt_model)
    classifier_layer = classifier_info["layers"][0]
    classifier_weight, classifier_bias = classifier_info["weights"][0]

    print(f"  Layer {layer_idx} MLP:")
    print(f"    Wi: {wi_info['in_features']} -> {wi_info['out_features']}")
    print(f"    Wo: {wo_info['in_features']} -> {wo_info['out_features']}")
    print(f"    Classifier: {classifier_layer['in_features']} -> {classifier_layer['out_features']}")

    return {
        "type": "mlp_classifier",
        "layer_idx": layer_idx,
        "hidden_size": wi_info["in_features"],  # 768
        "wi": {
            "in_features": wi_info["in_features"],
            "out_features": wi_info["out_features"],
            "weight": wi_info["weight"],
            "bias": wi_info["bias"]
        },
        "wo": {
            "in_features": wo_info["in_features"],
            "out_features": wo_info["out_features"],
            "weight": wo_info["weight"],
            "bias": wo_info["bias"]
        },
        "classifier": {
            "in_features": classifier_layer["in_features"],
            "out_features": classifier_layer["out_features"],
            "weight": classifier_weight,
            "bias": classifier_bias
        }
    }


def extract_multi_layer_mlp_and_classifier(pt_model, layer_indices: List[int]) -> Dict[str, Any]:
    """
    Extrait Wi, Wo, LayerNorm de PLUSIEURS layers + classifier pour calculer l'influence
    sur tous les paramètres MLP des layers sélectionnés en une seule passe.

    Architecture créée:
    Input -> LN -> MLP_layer[0] + residual -> LN -> MLP_layer[1] + residual -> ... -> LN -> Classifier

    Parameters
    ----------
    pt_model : PreTrainedModel
        Le modèle PyTorch
    layer_indices : List[int]
        Liste des indices des layers transformer (ex: [18, 19, 20, 21])
    """
    # Récupérer toutes les couches MLP
    all_mlp_layers = get_encoder_mlp_info(pt_model)

    # Récupérer le classifier
    classifier_info = get_classifier_info(pt_model)
    classifier_layer = classifier_info["layers"][0]
    classifier_weight, classifier_bias = classifier_info["weights"][0]

    # Extraire les LayerNorms pour chaque layer
    # ModernBERT utilise: model.layers[i].mlp_norm pour le pre-MLP LayerNorm
    layer_norms = {}
    final_ln = None

    for name, module in pt_model.named_modules():
        # LayerNorm avant MLP: model.layers.{i}.mlp_norm ou model.encoder.layer.{i}.output.LayerNorm
        if "mlp_norm" in name or "ffn_norm" in name:
            match = re.search(r"layers\.(\d+)", name)
            if match:
                idx = int(match.group(1))
                if idx in layer_indices:
                    layer_norms[idx] = {
                        "weight": module.weight.detach().cpu().numpy(),
                        "bias": module.bias.detach().cpu().numpy() if module.bias is not None else None
                    }
                    print(f"  LayerNorm pour layer {idx} trouvé: {name}")

        # LayerNorm finale (head_norm ou final_layer_norm)
        if name in ["model.head_norm", "model.final_layer_norm", "head_norm", "final_layer_norm"]:
            final_ln = {
                "weight": module.weight.detach().cpu().numpy(),
                "bias": module.bias.detach().cpu().numpy() if module.bias is not None else None
            }
            print(f"  LayerNorm finale trouvée: {name}")

    # Extraire les MLPs pour chaque layer demandé
    mlp_info_list = []
    for layer_idx in sorted(layer_indices):
        layer_mlp = [layer for layer in all_mlp_layers if layer["layer_idx"] == layer_idx]

        if len(layer_mlp) != 2:
            raise ValueError(f"Attendu 2 couches MLP (Wi, Wo) pour layer {layer_idx}, trouvé {len(layer_mlp)}")

        wi_info = next(layer for layer in layer_mlp if layer["which"] == "Wi")
        wo_info = next(layer for layer in layer_mlp if layer["which"] == "Wo")

        mlp_entry = {
            "layer_idx": layer_idx,
            "wi": {
                "in_features": wi_info["in_features"],
                "out_features": wi_info["out_features"],
                "weight": wi_info["weight"],
                "bias": wi_info["bias"]
            },
            "wo": {
                "in_features": wo_info["in_features"],
                "out_features": wo_info["out_features"],
                "weight": wo_info["weight"],
                "bias": wo_info["bias"]
            }
        }

        # Ajouter LayerNorm si trouvé
        if layer_idx in layer_norms:
            mlp_entry["layer_norm"] = layer_norms[layer_idx]

        mlp_info_list.append(mlp_entry)

        print(f"  Layer {layer_idx} MLP:")
        print(f"    Wi: {wi_info['in_features']} -> {wi_info['out_features']}")
        print(f"    Wo: {wo_info['in_features']} -> {wo_info['out_features']}")

    print(f"  Classifier: {classifier_layer['in_features']} -> {classifier_layer['out_features']}")

    # Déterminer la dimension d'entrée (première couche Wi)
    hidden_size = mlp_info_list[0]["wi"]["in_features"]

    return {
        "type": "multi_layer_mlp_classifier",
        "layer_indices": sorted(layer_indices),
        "hidden_size": hidden_size,
        "mlp_layers": mlp_info_list,
        "final_layer_norm": final_ln,
        "classifier": {
            "in_features": classifier_layer["in_features"],
            "out_features": classifier_layer["out_features"],
            "weight": classifier_weight,
            "bias": classifier_bias
        }
    }


def swiglu_activation(x):
    """
    SwiGLU activation: split input in half, apply SiLU to first half, multiply.

    x: tensor of shape (..., 2*intermediate_size)
    returns: tensor of shape (..., intermediate_size)
    """
    # Split en deux moitiés égales
    gate, up = tf.split(x, 2, axis=-1)
    # SiLU (Swish) sur la partie gate, puis multiplication
    return tf.nn.silu(gate) * up


def create_mlp_classifier_model(model_info: Dict[str, Any]) -> Model:
    """
    Crée un modèle TensorFlow avec: Wi -> SwiGLU -> Wo -> Classifier.

    Ce modèle permet de calculer l'influence sur toutes les couches Dense
    du MLP d'un layer transformer + le classifier.
    """
    hidden_size = model_info["hidden_size"]
    wi_info = model_info["wi"]
    wo_info = model_info["wo"]
    classifier = model_info["classifier"]

    # Input: hidden states (768)
    inputs = layers.Input(shape=(hidden_size,), name="embeddings")

    # Wi: 768 -> 2304
    x = layers.Dense(
        wi_info["out_features"],
        use_bias=wi_info["bias"] is not None,
        name="wi"
    )(inputs)

    # SwiGLU gating (pas de poids entraînables)
    x = layers.Lambda(swiglu_activation, name="swiglu")(x)

    # Wo: 1152 -> 768
    x = layers.Dense(
        wo_info["out_features"],
        use_bias=wo_info["bias"] is not None,
        name="wo"
    )(x)

    # Residual connection avec l'input
    x = layers.Add(name="residual")([inputs, x])

    # Classifier: 768 -> num_classes
    outputs = layers.Dense(
        classifier["out_features"],
        use_bias=classifier["bias"] is not None,
        name="classifier"
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="mlp_classifier_model")
    return model


def transfer_mlp_classifier_weights(model_info: Dict[str, Any], tf_model: Model) -> None:
    """
    Transfère les poids Wi, Wo et classifier de PyTorch vers TensorFlow.
    """
    print("Transfert des poids MLP + Classifier PyTorch -> TensorFlow...")

    # Wi
    wi_layer = tf_model.get_layer("wi")
    wi_weight = model_info["wi"]["weight"]
    wi_bias = model_info["wi"]["bias"]
    if wi_bias is not None:
        wi_layer.set_weights([wi_weight.T, wi_bias])
    else:
        wi_layer.set_weights([wi_weight.T])
    print(f"  wi: ({wi_weight.shape[1]}, {wi_weight.shape[0]})")

    # Wo
    wo_layer = tf_model.get_layer("wo")
    wo_weight = model_info["wo"]["weight"]
    wo_bias = model_info["wo"]["bias"]
    if wo_bias is not None:
        wo_layer.set_weights([wo_weight.T, wo_bias])
    else:
        wo_layer.set_weights([wo_weight.T])
    print(f"  wo: ({wo_weight.shape[1]}, {wo_weight.shape[0]})")

    # Classifier
    classifier_layer = tf_model.get_layer("classifier")
    classifier_weight = model_info["classifier"]["weight"]
    classifier_bias = model_info["classifier"]["bias"]
    if classifier_bias is not None:
        classifier_layer.set_weights([classifier_weight.T, classifier_bias])
    else:
        classifier_layer.set_weights([classifier_weight.T])
    print(f"  classifier: ({classifier_weight.shape[1]}, {classifier_weight.shape[0]})")

    print("Poids transférés avec succès!")


def create_multi_layer_mlp_model(model_info: Dict[str, Any]) -> Model:
    """
    Crée un modèle TensorFlow avec PLUSIEURS MLPs enchaînés + Classifier.

    Architecture:
    Input -> LN -> [Wi_L1 -> SwiGLU -> Wo_L1 + residual] -> LN -> [Wi_L2 -> ...] -> ... -> Classifier

    Chaque MLP a sa propre connexion résiduelle et LayerNorm pour stabilité.
    """
    hidden_size = model_info["hidden_size"]
    mlp_layers = model_info["mlp_layers"]
    classifier = model_info["classifier"]

    # Input: hidden states
    inputs = layers.Input(shape=(hidden_size,), name="embeddings")
    x = inputs

    # Pour chaque layer MLP
    for mlp_info in mlp_layers:
        layer_idx = mlp_info["layer_idx"]
        wi_info = mlp_info["wi"]
        wo_info = mlp_info["wo"]

        # Sauvegarder l'entrée pour la connexion résiduelle
        residual = x

        # LayerNorm avant le MLP (Pre-LN architecture)
        # trainable=False car KFAC ne supporte que les couches Dense
        x = layers.LayerNormalization(epsilon=1e-5, trainable=False, name=f"ln_{layer_idx}")(x)

        # Wi: 768 -> 2304
        x = layers.Dense(
            wi_info["out_features"],
            use_bias=wi_info["bias"] is not None,
            name=f"wi_{layer_idx}"
        )(x)

        # SwiGLU gating
        x = layers.Lambda(swiglu_activation, name=f"swiglu_{layer_idx}")(x)

        # Wo: 1152 -> 768
        x = layers.Dense(
            wo_info["out_features"],
            use_bias=wo_info["bias"] is not None,
            name=f"wo_{layer_idx}"
        )(x)

        # Connexion résiduelle
        x = layers.Add(name=f"residual_{layer_idx}")([residual, x])

    # LayerNorm finale avant le classifier
    # trainable=False car KFAC ne supporte que les couches Dense
    x = layers.LayerNormalization(epsilon=1e-5, trainable=False, name="ln_final")(x)

    # Classifier final
    outputs = layers.Dense(
        classifier["out_features"],
        use_bias=classifier["bias"] is not None,
        name="classifier"
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="multi_layer_mlp_model")
    return model


def transfer_multi_layer_weights(model_info: Dict[str, Any], tf_model: Model) -> None:
    """
    Transfère les poids de plusieurs MLPs + LayerNorms + classifier de PyTorch vers TensorFlow.
    """
    print("Transfert des poids Multi-Layer MLP + LayerNorm + Classifier PyTorch -> TensorFlow...")

    # Pour chaque MLP layer
    for mlp_info in model_info["mlp_layers"]:
        layer_idx = mlp_info["layer_idx"]

        # LayerNorm (si présent dans les poids PyTorch)
        if "layer_norm" in mlp_info:
            try:
                ln_layer = tf_model.get_layer(f"ln_{layer_idx}")
                ln_weight = mlp_info["layer_norm"]["weight"]
                ln_bias = mlp_info["layer_norm"]["bias"]
                if ln_bias is not None:
                    ln_layer.set_weights([ln_weight, ln_bias])
                else:
                    ln_layer.set_weights([ln_weight, np.zeros_like(ln_weight)])
                print(f"  ln_{layer_idx}: ({ln_weight.shape[0]},)")
            except ValueError:
                print(f"  ln_{layer_idx}: non trouvé dans le modèle TF")

        # Wi
        wi_layer = tf_model.get_layer(f"wi_{layer_idx}")
        wi_weight = mlp_info["wi"]["weight"]
        wi_bias = mlp_info["wi"]["bias"]
        if wi_bias is not None:
            wi_layer.set_weights([wi_weight.T, wi_bias])
        else:
            wi_layer.set_weights([wi_weight.T])
        print(f"  wi_{layer_idx}: ({wi_weight.shape[1]}, {wi_weight.shape[0]})")

        # Wo
        wo_layer = tf_model.get_layer(f"wo_{layer_idx}")
        wo_weight = mlp_info["wo"]["weight"]
        wo_bias = mlp_info["wo"]["bias"]
        if wo_bias is not None:
            wo_layer.set_weights([wo_weight.T, wo_bias])
        else:
            wo_layer.set_weights([wo_weight.T])
        print(f"  wo_{layer_idx}: ({wo_weight.shape[1]}, {wo_weight.shape[0]})")

    # LayerNorm finale
    if model_info.get("final_layer_norm"):
        try:
            ln_final = tf_model.get_layer("ln_final")
            ln_weight = model_info["final_layer_norm"]["weight"]
            ln_bias = model_info["final_layer_norm"]["bias"]
            if ln_bias is not None:
                ln_final.set_weights([ln_weight, ln_bias])
            else:
                ln_final.set_weights([ln_weight, np.zeros_like(ln_weight)])
            print(f"  ln_final: ({ln_weight.shape[0]},)")
        except ValueError:
            print("  ln_final: non trouvé dans le modèle TF")

    # Classifier
    classifier_layer = tf_model.get_layer("classifier")
    classifier_weight = model_info["classifier"]["weight"]
    classifier_bias = model_info["classifier"]["bias"]
    if classifier_bias is not None:
        classifier_layer.set_weights([classifier_weight.T, classifier_bias])
    else:
        classifier_layer.set_weights([classifier_weight.T])
    print(f"  classifier: ({classifier_weight.shape[1]}, {classifier_weight.shape[0]})")

    print("Poids transférés avec succès!")


def extract_intermediate_embeddings(
    pt_model,
    tokenizer,
    texts: list,
    max_len: int,
    layer_idx: int,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extrait les embeddings à la sortie d'un layer transformer spécifique.

    Utilise output_hidden_states=True pour récupérer les hidden states
    de chaque couche, puis extrait le [CLS] token du layer demandé.

    Note: On récupère la SORTIE du layer (après attention + MLP + residual),
    pas l'état intermédiaire après gating. C'est une approximation mais
    plus fiable que les hooks qui ne fonctionnent pas avec torch.compile.

    Parameters
    ----------
    layer_idx : int
        Index du layer transformer (0 à 21 pour ModernBERT)
    """
    pt_model.eval()
    pt_model.to(DEVICE)
    all_embeddings = []

    print(f"  Extraction via output_hidden_states (layer {layer_idx})")

    for i in tqdm(range(0, len(texts), batch_size), desc="Extraction embeddings"):
        batch_texts = texts[i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = pt_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # hidden_states est un tuple de (n_layers + 1) tenseurs
            # Index 0 = embeddings, Index 1 = sortie layer 0, ..., Index 22 = sortie layer 21
            hidden_states = outputs.hidden_states

            if i == 0:
                print(f"  Nombre de hidden states: {len(hidden_states)}")
                print(f"  Shape de chaque état: {hidden_states[0].shape}")

            # Récupérer la sortie du layer demandé (+1 car index 0 = embeddings)
            # Pour layer_idx=21, on prend hidden_states[22] (la sortie du dernier layer)
            layer_output = hidden_states[layer_idx + 1]

            # Prendre le token [CLS] (position 0)
            cls_embeddings = layer_output[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.concatenate(all_embeddings, axis=0)


def main():
    print("="*60)
    print("ANALYSE D'INFLUENCE (KFAC) - Transformer sur NOTAM")
    print(f"Mode: {INFLUENCE_MODE}")
    print("="*60)

    # Charger le dataset
    print("\nChargement du dataset...")
    dataset = load_dataset(DATASET_NAME)
    train_ds_hf = dataset["train"]
    test_ds_hf = dataset["test"]

    # Récupérer les labels
    label_names = list(set(train_ds_hf["label"]))
    label_names.sort()
    label2id = {name: i for i, name in enumerate(label_names)}
    num_classes = len(label_names)

    print(f"Classes ({num_classes}): {label_names}")

    # Limiter le nombre d'exemples si spécifié
    if NUM_TRAIN_SAMPLES is not None:
        train_ds_hf = train_ds_hf.select(range(min(NUM_TRAIN_SAMPLES, len(train_ds_hf))))
    if NUM_TEST_SAMPLES is not None:
        test_ds_hf = test_ds_hf.select(range(min(NUM_TEST_SAMPLES, len(test_ds_hf))))

    print(f"Train: {len(train_ds_hf)} exemples, Test: {len(test_ds_hf)} exemples")

    # Charger le modèle PyTorch
    print(f"\nChargement du modèle: {MODEL_NAME}")
    pt_model, tokenizer = load_transformer_model(MODEL_NAME)

    # Convertir les labels en indices
    train_labels = np.array([label2id[lbl] for lbl in train_ds_hf["label"]], dtype=np.int32)
    test_labels = np.array([label2id[lbl] for lbl in test_ds_hf["label"]], dtype=np.int32)

    train_texts = list(train_ds_hf["text"])
    test_texts = list(test_ds_hf["text"])

    if INFLUENCE_MODE == "classifier_only":
        # Mode classifier uniquement
        print(f"\nExtraction des couches ({INFLUENCE_MODE})...")
        model_info = get_classifier_info(pt_model)

        print(f"Type: {model_info['type']}")
        print("Couches:")
        for layer in model_info.get('layers', []):
            print(f"  {layer['name']}: {layer['in_features']} -> {layer['out_features']}")

        # Extraire les embeddings finaux
        print("\nExtraction des embeddings du transformer...")
        train_embeddings = extract_embeddings_from_transformer(
            pt_model, tokenizer, train_texts, MAX_LEN, batch_size=32
        )
        test_embeddings = extract_embeddings_from_transformer(
            pt_model, tokenizer, test_texts, MAX_LEN, batch_size=32
        )

        print(f"Train embeddings: {train_embeddings.shape}")
        print(f"Test embeddings: {test_embeddings.shape}")

        # Créer et configurer le modèle TF
        tf_model = create_tf_model(model_info)
        tf_model.summary()
        transfer_weights_to_tf(model_info, tf_model)

        tf_model.compile(
            optimizer="adam",
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        # Vérification
        test_ds = prepare_tf_dataset(test_embeddings, test_labels, BATCH_SIZE)
        loss, acc = tf_model.evaluate(test_ds, verbose=0)
        print(f"Vérification TF - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        # Calculer l'influence
        results = compute_influence_with_kfac(
            tf_model, train_embeddings, train_labels,
            test_embeddings, test_labels,
            batch_size=BATCH_SIZE, batch_size_hessian=BATCH_SIZE_HESSIAN,
            n_samples_hessian=N_SAMPLES_HESSIAN, top_k=TOP_K
        )

        # Afficher les résultats
        display_results(results, train_ds_hf, test_ds_hf, train_embeddings, label_names, top_k=TOP_K)

    else:
        # Mode with_encoder_mlp : tous les MLPs des layers sélectionnés en une seule passe
        print(f"\n{'='*60}")
        print(f"MULTI-LAYER MLP: Layers {ENCODER_LAYERS}")
        print(f"{'='*60}")

        # Debug: afficher les modules LayerNorm
        debug_print_model_modules(pt_model, "norm")
        debug_print_model_modules(pt_model, "layer.21")

        # Extraire TOUS les MLPs (Wi, Wo) des layers sélectionnés + classifier
        print(f"\nExtraction des couches MLP + Classifier pour layers {ENCODER_LAYERS}...")
        model_info = extract_multi_layer_mlp_and_classifier(pt_model, ENCODER_LAYERS)

        print(f"\nType: {model_info['type']}")
        print(f"Layers: {model_info['layer_indices']}")
        print("Architecture:")
        print("  Input (768)")
        for mlp in model_info['mlp_layers']:
            print(f"    -> Wi_{mlp['layer_idx']} -> SwiGLU -> Wo_{mlp['layer_idx']} + residual")
        print("    -> Classifier")

        # Nombre total de couches Dense
        num_dense_layers = len(model_info['mlp_layers']) * 2 + 1  # Wi + Wo pour chaque layer + classifier
        print(f"\nNombre de couches Dense: {num_dense_layers}")

        # Extraire les embeddings à l'entrée du premier layer MLP sélectionné
        # Pour layer 18, on prend hidden_states[18] = sortie du layer 17
        first_layer = min(ENCODER_LAYERS)
        input_layer_idx = first_layer - 1 if first_layer > 0 else 0
        print(f"\nExtraction des embeddings (entrée du layer {first_layer}, hidden_states[{input_layer_idx + 1}])...")
        train_embeddings = extract_intermediate_embeddings(
            pt_model, tokenizer, train_texts, MAX_LEN, input_layer_idx, batch_size=32
        )
        test_embeddings = extract_intermediate_embeddings(
            pt_model, tokenizer, test_texts, MAX_LEN, input_layer_idx, batch_size=32
        )

        print(f"Train embeddings: {train_embeddings.shape}")
        print(f"Test embeddings: {test_embeddings.shape}")

        # Créer le modèle TF avec tous les MLPs enchaînés
        print("\nCréation du modèle TensorFlow (Multi-Layer MLP + Classifier)...")
        tf_model = create_multi_layer_mlp_model(model_info)
        tf_model.summary()
        transfer_multi_layer_weights(model_info, tf_model)

        tf_model.compile(
            optimizer="adam",
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        # Vérification
        test_ds = prepare_tf_dataset(test_embeddings, test_labels, BATCH_SIZE)
        loss, acc = tf_model.evaluate(test_ds, verbose=0)
        print(f"Vérification TF - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        # Calculer l'influence sur TOUS les paramètres des MLPs sélectionnés
        print(f"\nCalcul de l'influence sur {num_dense_layers} couches Dense...")
        results = compute_influence_with_kfac(
            tf_model, train_embeddings, train_labels,
            test_embeddings, test_labels,
            batch_size=BATCH_SIZE, batch_size_hessian=BATCH_SIZE_HESSIAN,
            n_samples_hessian=N_SAMPLES_HESSIAN, top_k=TOP_K
        )

        # Afficher les résultats
        print(f"\n--- Résultats pour Layers {ENCODER_LAYERS} ---")
        display_results(results, train_ds_hf, test_ds_hf, train_embeddings, label_names, top_k=TOP_K)

    print("\n" + "="*60)
    print("Analyse terminée!")
    print("="*60)


if __name__ == "__main__":
    main()
