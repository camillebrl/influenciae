"""
Script de validation KFAC et EKFAC vs reference implementations.

Ce script compare les implementations KFAC et EKFAC de influenciae avec:
1. L'implementation ExactIHVP (reference exacte)
2. l'implementation de nngeometry

Tests KFAC:
- Test 1: Facteurs A et G corrects + Pi-damping
- Test 2: KFAC IHVP vs ExactIHVP (reference)
- Test 3: HVP(IHVP(v)) ≈ v (verification de l'inverse)
- Test 3a: Comparaison directe avec nngeometry PMatKFAC
- Test 3b: Verification de la formule de solve KFAC
- Test 4: Layout des parametres
- Test 5: Scores d'influence

Tests EKFAC:
- Test 6: Eigendecomposition des facteurs A et G
- Test 7: EKFAC IHVP vs ExactIHVP (reference)
- Test 8: Comparaison directe avec nngeometry PMatEKFAC
- Test 9: Verification de la formule de solve EKFAC
- Test 10: Mise a jour des valeurs propres (eigenvalue correction)

Test Scores d'Influence:
- Test 11: Comparaison des scores d'influence entre influenciae et nngeometry
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add nngeometry to path
import sys
sys.path.insert(0, '/home/camil/Documents/nngeometry')

from deel.influenciae.common import InfluenceModel, KFACIHVP, EKFACIHVP, ExactIHVP
from deel.influenciae.influence import FirstOrderInfluenceCalculator

# Import nngeometry
from nngeometry.metrics import FIM
from nngeometry.object.pspace import PMatKFAC, PMatEKFAC
from nngeometry.object import PVector
from nngeometry.layercollection import LayerCollection


def create_simple_mlp(input_dim: int = 10, hidden_dim: int = 8, output_dim: int = 3):
    """Cree un MLP simple pour les tests."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim, activation="relu", name="hidden")(inputs)
    outputs = layers.Dense(output_dim, activation=None, name="output")(x)
    return Model(inputs=inputs, outputs=outputs)


def create_test_data(n_samples: int = 100, input_dim: int = 10, n_classes: int = 3, batch_size: int = 16):
    """Genere des donnees synthetiques pour les tests."""
    np.random.seed(42)
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, n_classes, size=(n_samples,)).astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size)
    return dataset, X, y


def process_batch(batch):
    """Fonction de preprocessing pour InfluenceModel."""
    inputs, labels = batch
    sample_weights = tf.ones_like(labels, dtype=tf.float32)
    return inputs, labels, sample_weights


def compute_kfac_factors_manual(model, dataset, layer_name="output"):
    """
    Calcule manuellement les facteurs KFAC A et G selon les formules de reference.

    A = (1/N) * sum(a @ a.T)  ou a est l'activation (augmentee avec 1 si bias)
    G = (1/N) * sum(g @ g.T)  ou g est le gradient de la loss par rapport au pre-activation
    """
    layer = model.get_layer(layer_name)
    has_bias = layer.bias is not None

    # Creer un modele pour extraire les activations
    activation_model = Model(inputs=model.input, outputs=model.get_layer("hidden").output)

    A_sum = None
    G_sum = None
    n_samples = 0

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)

    for batch in dataset:
        inputs, labels = batch
        batch_size = inputs.shape[0]
        n_samples += batch_size

        # Calcul de A: covariance des activations
        activations = activation_model(inputs).numpy()  # (batch, hidden_dim)

        if has_bias:
            # Augmenter avec une colonne de 1
            ones = np.ones((batch_size, 1), dtype=np.float32)
            activations_aug = np.concatenate([activations, ones], axis=1)
        else:
            activations_aug = activations

        # A = sum(a @ a.T) pour chaque sample
        A_batch = np.einsum('bi,bj->ij', activations_aug, activations_aug)

        # Calcul de G: covariance des gradients
        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss = loss_fn(labels, logits)

        # Gradient par rapport aux logits (pre-softmax)
        grad_logits = tape.gradient(loss, logits).numpy()  # (batch, output_dim)

        # G = sum(g @ g.T) pour chaque sample
        G_batch = np.einsum('bi,bj->ij', grad_logits, grad_logits)

        if A_sum is None:
            A_sum = A_batch
            G_sum = G_batch
        else:
            A_sum += A_batch
            G_sum += G_batch

    A = A_sum / n_samples
    G = G_sum / n_samples

    return A, G


def compute_pi_damping(A, G):
    """Calcule le facteur pi pour le damping adaptatif."""
    tr_A = np.trace(A)
    tr_G = np.trace(G)
    dim_A = A.shape[0]
    dim_G = G.shape[0]

    pi = np.sqrt((tr_A / tr_G) * (dim_G / dim_A))
    return pi


def kfac_solve_manual(A, G, V, damping=1e-4, use_pi=True):
    """
    Resout le systeme KFAC manuellement.

    KFAC approxime H^{-1} @ v comme:
    (G^{-1} @ V @ A^{-1}) avec V reshape en (out_dim, in_dim+bias)

    Avec pi-damping:
    A_reg = A + sqrt(damping) * pi * I
    G_reg = G + sqrt(damping) / pi * I
    """
    if use_pi:
        pi = compute_pi_damping(A, G)
    else:
        pi = 1.0

    sqrt_damp = np.sqrt(damping)
    A_reg = A + sqrt_damp * pi * np.eye(A.shape[0])
    G_reg = G + sqrt_damp / pi * np.eye(G.shape[0])

    out_dim = G.shape[0]
    in_dim = A.shape[0]

    # V doit etre reshape en (out_dim, in_dim)
    V_mat = V.reshape(out_dim, in_dim)

    # Solve: X = G_reg^{-1} @ V @ A_reg^{-1}
    # Equivalent a: lstsq(G_reg, V), puis lstsq(A_reg, result.T).T
    X = np.linalg.lstsq(G_reg, V_mat, rcond=None)[0]
    result = np.linalg.lstsq(A_reg, X.T, rcond=None)[0].T

    return result.flatten()


def test_kfac_factors():
    """Test 1: Verifie que les facteurs A et G sont correctement calcules."""
    print("\n" + "="*60)
    print("TEST 1: Verification des facteurs A et G")
    print("="*60)

    # Setup
    model = create_simple_mlp(input_dim=10, hidden_dim=8, output_dim=3)
    dataset, _, _ = create_test_data(n_samples=64, input_dim=10, batch_size=16)

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    influence_model = InfluenceModel(
        model, start_layer=1, last_layer=-1,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch
    )

    # KFAC de influenciae
    kfac_ihvp = KFACIHVP(model=influence_model, train_dataset=dataset)

    # Facteurs manuels
    A_manual, G_manual = compute_kfac_factors_manual(model, dataset, layer_name="output")

    # Recuperer les facteurs de KFACIHVP
    # Layer index pour la couche output
    layer_indices = list(kfac_ihvp.kfac_blocks.keys())
    print(f"Layers tracked: {layer_indices}")

    for layer_idx in layer_indices:
        A_kfac, G_kfac = kfac_ihvp.kfac_blocks[layer_idx]
        layer_info = kfac_ihvp.layer_info[layer_idx]
        print(f"\nLayer {layer_idx} ({layer_info['layer'].name}):")
        print(f"  A shape: KFAC={A_kfac.shape}, Manual={A_manual.shape}")
        print(f"  G shape: KFAC={G_kfac.shape}, Manual={G_manual.shape}")

        if layer_info['layer'].name == "output":
            # Comparer les facteurs
            A_diff = np.abs(A_kfac.numpy() - A_manual).max()
            G_diff = np.abs(G_kfac.numpy() - G_manual).max()

            print(f"  A max diff: {A_diff:.6e}")
            print(f"  G max diff: {G_diff:.6e}")

            # Verifier les traces pour pi
            pi_manual = compute_pi_damping(A_manual, G_manual)
            pi_kfac = compute_pi_damping(A_kfac.numpy(), G_kfac.numpy())
            print(f"  Pi (manual): {pi_manual:.6f}")
            print(f"  Pi (KFAC):   {pi_kfac:.6f}")

            if A_diff < 1e-4 and G_diff < 1e-4:
                print("  [PASS] Facteurs A et G corrects!")
            else:
                print("  [FAIL] Difference significative dans les facteurs")

    return kfac_ihvp, influence_model, dataset


def test_kfac_vs_exact():
    """Test 2: Compare KFAC IHVP avec ExactIHVP (reference)."""
    print("\n" + "="*60)
    print("TEST 2: KFAC vs ExactIHVP")
    print("="*60)

    # Setup avec petit modele
    model = create_simple_mlp(input_dim=8, hidden_dim=6, output_dim=3)
    dataset, X, y = create_test_data(n_samples=32, input_dim=8, n_classes=3, batch_size=8)

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    influence_model = InfluenceModel(
        model, start_layer=1, last_layer=-1,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch
    )

    print(f"Nombre de parametres: {influence_model.nb_params}")

    # KFAC IHVP
    print("Calcul de KFACIHVP...")
    kfac_ihvp = KFACIHVP(model=influence_model, train_dataset=dataset)

    # ExactIHVP (utilise la vraie inverse Hessienne)
    print("Calcul de ExactIHVP (peut prendre du temps)...")
    exact_ihvp_obj = ExactIHVP(model=influence_model, train_dataset=dataset)
    print(f"  Inverse Hessian shape: {exact_ihvp_obj.inv_hessian.shape}")

    # Comparer sur un batch de test
    test_dataset = tf.data.Dataset.from_tensor_slices((X[:4], y[:4])).batch(4)

    print("\nComparaison des IHVP:")
    for batch in test_dataset:
        # Gradient du batch
        grads = influence_model.batch_jacobian_tensor(batch).numpy()
        print(f"  Gradients shape: {grads.shape}")

        # IHVP avec ExactIHVP: retourne (nb_params, batch)
        exact_result = exact_ihvp_obj._compute_ihvp_single_batch(batch).numpy()

        # IHVP KFAC: retourne (nb_params, batch)
        kfac_result = kfac_ihvp._compute_ihvp_single_batch(batch).numpy()

        print(f"  ExactIHVP shape: {exact_result.shape} (nb_params, batch)")
        print(f"  KFAC IHVP shape: {kfac_result.shape}")

        # Calculer la correlation pour chaque sample
        batch_size = grads.shape[0]
        for i in range(min(4, batch_size)):
            exact_vec = exact_result[:, i].flatten()
            kfac_vec = kfac_result[:, i].flatten()

            # Normaliser
            exact_norm = exact_vec / (np.linalg.norm(exact_vec) + 1e-10)
            kfac_norm = kfac_vec / (np.linalg.norm(kfac_vec) + 1e-10)

            cos_sim = np.dot(exact_norm, kfac_norm)
            rel_error = np.linalg.norm(exact_vec - kfac_vec) / (np.linalg.norm(exact_vec) + 1e-10)

            print(f"  Sample {i}: cos_sim={cos_sim:.4f}, rel_error={rel_error:.4f}")

        break

    # Note sur les resultats attendus
    print("\n[INFO] Cosine similarity attendue pour KFAC:")
    print("  - KFAC est une APPROXIMATION, donc cos_sim < 1.0 est normal")
    print("  - cos_sim > 0.5 indique une bonne approximation")
    print("  - cos_sim proche de 1.0 est excellent")

    return kfac_ihvp, exact_ihvp_obj


def compute_hessian_manual(influence_model, dataset):
    """Calcule la Hessienne manuellement en utilisant les Jacobiens."""
    n_params = influence_model.nb_params

    # Accumuler les Hessiennes par batch
    hessian_sum = np.zeros((n_params, n_params), dtype=np.float32)
    n_samples = 0

    for batch in dataset:
        inputs, labels = batch
        batch_size = inputs.shape[0]

        # Pour chaque sample, calculer la Hessienne de la loss
        for i in range(batch_size):
            single_input = inputs[i:i+1]
            single_label = labels[i:i+1]

            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape() as tape1:
                    # Forward
                    logits = influence_model.model(single_input)
                    loss = influence_model.loss_function(single_label, logits)
                    loss = tf.reduce_mean(loss)

                # Premier gradient
                grads = tape1.gradient(loss, influence_model.weights)
                grads_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

            # Second gradient (Hessienne)
            hess_rows = []
            for j in range(n_params):
                grad_j = tape2.gradient(grads_flat[j], influence_model.weights)
                if grad_j[0] is None:
                    hess_row = np.zeros(n_params, dtype=np.float32)
                else:
                    hess_row = tf.concat([tf.reshape(g, [-1]) for g in grad_j], axis=0).numpy()
                hess_rows.append(hess_row)

            del tape2
            hessian_sample = np.stack(hess_rows)
            hessian_sum += hessian_sample
            n_samples += 1

    return hessian_sum / n_samples


def test_hvp_ihvp_identity():
    """Test 3: Verifie que HVP(IHVP(v)) ≈ v en utilisant ExactIHVP.

    Ce test verifie que:
    1. ExactIHVP calcule correctement H^{-1} @ v
    2. Si on applique H @ (H^{-1} @ v), on retrouve v
    """
    print("\n" + "="*60)
    print("TEST 3: HVP(IHVP(v)) ≈ v avec ExactIHVP")
    print("="*60)

    model = create_simple_mlp(input_dim=8, hidden_dim=6, output_dim=3)
    dataset, X, y = create_test_data(n_samples=32, input_dim=8, n_classes=3, batch_size=8)

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    influence_model = InfluenceModel(
        model, start_layer=1, last_layer=-1,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch
    )

    # KFAC et ExactIHVP
    print("Calcul de KFACIHVP...")
    kfac_ihvp = KFACIHVP(model=influence_model, train_dataset=dataset)

    print("Calcul de ExactIHVP...")
    exact_ihvp = ExactIHVP(model=influence_model, train_dataset=dataset)

    # Recuperer la Hessienne depuis ExactIHVP (on peut la recalculer a partir de inv_hessian)
    # Note: ExactIHVP stocke inv_hessian, pas hessian
    inv_hessian = exact_ihvp.inv_hessian.numpy()
    print(f"  Inverse Hessian shape: {inv_hessian.shape}")

    # Recalculer la Hessienne (ou utiliser pinv de l'inverse)
    hessian = np.linalg.pinv(inv_hessian)
    print(f"  Hessian (from pinv) shape: {hessian.shape}")

    # Prendre un vecteur v (gradient d'un sample)
    test_batch = next(iter(dataset))
    v = influence_model.batch_jacobian_tensor(test_batch)[0:1]  # Premier sample
    v = tf.reshape(v, (-1,))

    print(f"\nVecteur v shape: {v.shape}")
    print(f"Vecteur v norm: {tf.norm(v).numpy():.6f}")

    # IHVP avec ExactIHVP
    exact_ihvp_result = exact_ihvp._compute_ihvp_single_batch(test_batch)[:, 0]
    print(f"ExactIHVP result shape: {exact_ihvp_result.shape}")
    print(f"ExactIHVP norm: {tf.norm(exact_ihvp_result).numpy():.6f}")

    # IHVP avec KFAC
    kfac_ihvp_result = kfac_ihvp._compute_ihvp_single_batch(test_batch)[:, 0]
    print(f"KFAC IHVP shape: {kfac_ihvp_result.shape}")
    print(f"KFAC IHVP norm: {tf.norm(kfac_ihvp_result).numpy():.6f}")

    # HVP avec la Hessienne: H @ (H^{-1} @ v) devrait ≈ v
    hvp_exact = hessian @ exact_ihvp_result.numpy()
    hvp_kfac = hessian @ kfac_ihvp_result.numpy()

    print(f"\nHVP(ExactIHVP) norm: {np.linalg.norm(hvp_exact):.6f}")
    print(f"HVP(KFAC_IHVP) norm: {np.linalg.norm(hvp_kfac):.6f}")

    # Comparer v et HVP(IHVP(v))
    v_np = v.numpy()

    # Pour ExactIHVP: devrait etre parfait
    cos_sim_exact = np.dot(v_np, hvp_exact) / (np.linalg.norm(v_np) * np.linalg.norm(hvp_exact) + 1e-10)
    # Pour KFAC: approximation
    cos_sim_kfac = np.dot(v_np, hvp_kfac) / (np.linalg.norm(v_np) * np.linalg.norm(hvp_kfac) + 1e-10)

    print(f"\nResultats:")
    print(f"  ExactIHVP: cos_sim(v, H @ H^-1 @ v) = {cos_sim_exact:.6f}")
    print(f"  KFAC:      cos_sim(v, H @ KFAC^-1 @ v) = {cos_sim_kfac:.6f}")

    if cos_sim_exact > 0.99:
        print("  [PASS] ExactIHVP: H @ H^-1 @ v ≈ v (attendu)")
    else:
        print("  [WARN] ExactIHVP diverge - probleme numerique (Hessienne singuliere?)")

    if cos_sim_kfac > 0.5:
        print("  [PASS] KFAC approxime bien l'inverse Hessienne")
    else:
        print("  [INFO] KFAC: approximation moderee (normal pour KFAC)")


def test_kfac_vs_nngeometry():
    """Test 3a: Compare directement avec nngeometry PMatKFAC."""
    print("\n" + "="*60)
    print("TEST 3a: KFAC vs nngeometry (comparaison directe)")
    print("="*60)

    # Setup: memes donnees pour TF et PyTorch
    np.random.seed(42)
    n_samples = 64
    input_dim = 8
    hidden_dim = 6
    output_dim = 3
    batch_size = 16

    X_np = np.random.randn(n_samples, input_dim).astype(np.float32)
    y_np = np.random.randint(0, output_dim, size=(n_samples,)).astype(np.int64)

    # === TensorFlow model ===
    tf_model = create_simple_mlp(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Get TF weights
    tf_hidden_w = tf_model.get_layer("hidden").kernel.numpy()  # (in, hidden)
    tf_hidden_b = tf_model.get_layer("hidden").bias.numpy()
    tf_output_w = tf_model.get_layer("output").kernel.numpy()  # (hidden, out)
    tf_output_b = tf_model.get_layer("output").bias.numpy()

    # === PyTorch model with SAME weights ===
    class PyTorchMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            return self.output(x)

    pt_model = PyTorchMLP()

    # Transfer weights: Keras (in, out) -> PyTorch (out, in)
    pt_model.hidden.weight.data = torch.tensor(tf_hidden_w.T)
    pt_model.hidden.bias.data = torch.tensor(tf_hidden_b)
    pt_model.output.weight.data = torch.tensor(tf_output_w.T)
    pt_model.output.bias.data = torch.tensor(tf_output_b)

    # Verify same forward pass
    x_test = torch.tensor(X_np[:2])
    tf_out = tf_model(X_np[:2]).numpy()
    pt_out = pt_model(x_test).detach().numpy()
    print(f"Forward pass match: {np.allclose(tf_out, pt_out, atol=1e-5)}")

    # === Compute KFAC with TensorFlow/influenciae ===
    tf_dataset = tf.data.Dataset.from_tensor_slices((X_np, y_np.astype(np.int32))).batch(batch_size)
    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    influence_model = InfluenceModel(
        tf_model, start_layer=1, last_layer=-1,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch
    )
    kfac_tf = KFACIHVP(model=influence_model, train_dataset=tf_dataset)

    print(f"\nTF KFAC computed. Layers: {len(kfac_tf.kfac_blocks)}")
    for idx, (A, G) in kfac_tf.kfac_blocks.items():
        info = kfac_tf.layer_info[idx]
        print(f"  Layer {idx} ({info['layer'].name}): A={A.shape}, G={G.shape}")

    # === Compute KFAC with nngeometry ===
    pt_loader = DataLoader(TensorDataset(torch.tensor(X_np), torch.tensor(y_np)), batch_size=batch_size)

    # Use FIM with classif_logits variant
    F_kfac = FIM(
        model=pt_model,
        loader=pt_loader,
        representation=PMatKFAC,
        variant='classif_logits',
        device='cpu'
    )

    print(f"\nnngeometry KFAC computed. Layers: {len(F_kfac.data)}")
    for layer_id, (a_nn, g_nn) in F_kfac.data.items():
        print(f"  Layer {layer_id}: A={a_nn.shape}, G={g_nn.shape}")

    # === Compare factors A and G ===
    print("\n--- Comparaison des facteurs A et G ---")

    # Get output layer factors
    # nngeometry: layer_id is the module name
    nn_layer_ids = list(F_kfac.data.keys())
    print(f"nngeometry layer IDs: {nn_layer_ids}")

    # Compare output layer (last layer)
    output_layer_id = nn_layer_ids[-1]  # 'output'
    A_nn, G_nn = F_kfac.data[output_layer_id]
    A_nn = A_nn.numpy()
    G_nn = G_nn.numpy()

    # TF: output layer is index 1
    A_tf = kfac_tf.kfac_blocks[1][0].numpy()
    G_tf = kfac_tf.kfac_blocks[1][1].numpy()

    print("\nOutput layer:")
    print(f"  A_tf shape: {A_tf.shape}, A_nn shape: {A_nn.shape}")
    print(f"  G_tf shape: {G_tf.shape}, G_nn shape: {G_nn.shape}")

    # Note: nngeometry may use different scaling
    # Compare normalized versions
    A_tf_norm = A_tf / (np.linalg.norm(A_tf) + 1e-10)
    A_nn_norm = A_nn / (np.linalg.norm(A_nn) + 1e-10)
    G_tf_norm = G_tf / (np.linalg.norm(G_tf) + 1e-10)
    G_nn_norm = G_nn / (np.linalg.norm(G_nn) + 1e-10)

    cos_sim_A = np.sum(A_tf_norm * A_nn_norm)
    cos_sim_G = np.sum(G_tf_norm * G_nn_norm)

    print(f"  A cosine similarity: {cos_sim_A:.6f}")
    print(f"  G cosine similarity: {cos_sim_G:.6f}")

    # === Compare IHVP/solve ===
    print("\n--- Comparaison du solve KFAC ---")

    # Create a gradient vector
    x_sample = torch.tensor(X_np[:1], requires_grad=True)
    y_sample = torch.tensor(y_np[:1])

    pt_model.zero_grad()
    logits = pt_model(x_sample)
    loss = nn.CrossEntropyLoss()(logits, y_sample)
    loss.backward()

    # Get gradient as PVector for nngeometry
    layer_collection = LayerCollection.from_model(pt_model)
    grad_dict = {}
    for name, param in pt_model.named_parameters():
        layer_name = name.split('.')[0]  # 'hidden' or 'output'
        param_type = name.split('.')[1]  # 'weight' or 'bias'
        if layer_name not in grad_dict:
            grad_dict[layer_name] = [None, None]
        if param_type == 'weight':
            grad_dict[layer_name][0] = param.grad.clone()
        else:
            grad_dict[layer_name][1] = param.grad.clone()

    # Convert to proper format
    grad_pvector_dict = {}
    for layer_id in F_kfac.data.keys():
        w_grad = grad_dict[layer_id][0]  # (out, in)
        b_grad = grad_dict[layer_id][1]  # (out,)
        grad_pvector_dict[layer_id] = (w_grad, b_grad)

    grad_pvector = PVector(layer_collection=layer_collection, dict_repr=grad_pvector_dict)

    # Solve with nngeometry
    damping = kfac_tf.regul
    solved_nn = F_kfac.solvePVec(grad_pvector, regul=damping, use_pi=True)

    # Get the output layer result
    solved_nn_output = solved_nn.to_dict()[output_layer_id]
    solved_nn_w = solved_nn_output[0].numpy().flatten()
    solved_nn_b = solved_nn_output[1].numpy().flatten()
    solved_nn_vec = np.concatenate([solved_nn_w, solved_nn_b])

    print(f"nngeometry solve output shape: {solved_nn_vec.shape}")

    # Solve with TF KFAC
    tf_batch = (tf.constant(X_np[:1]), tf.constant(y_np[:1].astype(np.int32)))
    ihvp_tf = kfac_tf._compute_ihvp_single_batch(tf_batch)[:, 0].numpy()

    # Extract output layer block
    param_start = kfac_tf.layer_info[0]['n_params']
    n_params_output = kfac_tf.layer_info[1]['n_params']
    ihvp_tf_output = ihvp_tf[param_start:param_start + n_params_output]

    print(f"TF KFAC solve output shape: {ihvp_tf_output.shape}")

    # Compare
    # Note: ordering might be different
    cos_sim_solve = np.dot(ihvp_tf_output, solved_nn_vec) / (
        np.linalg.norm(ihvp_tf_output) * np.linalg.norm(solved_nn_vec) + 1e-10)

    print("\nSolve comparison:")
    print(f"  TF first 5: {ihvp_tf_output[:5]}")
    print(f"  NN first 5: {solved_nn_vec[:5]}")
    print(f"  Cosine similarity: {cos_sim_solve:.6f}")

    if abs(cos_sim_solve) > 0.95:
        print("  [PASS] TF KFAC matches nngeometry")
    else:
        print("  [WARN] Difference between TF KFAC and nngeometry")
        print("  Note: May be due to different ordering (weight/bias) or scaling")

    return kfac_tf, F_kfac


def test_kfac_solve_formula():
    """Test 3b: Verifie la formule de solve KFAC."""
    print("\n" + "="*60)
    print("TEST 3b: Formule KFAC solve")
    print("="*60)

    # Setup: utiliser TOUTES les couches pour eviter le bug de layer_info
    # (KFAC._extract_layer_info() ne respecte pas start_layer/last_layer)
    model = create_simple_mlp(input_dim=6, hidden_dim=4, output_dim=3)
    dataset, X, y = create_test_data(n_samples=32, input_dim=6, n_classes=3, batch_size=8)

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    influence_model = InfluenceModel(
        model, start_layer=1, last_layer=-1,  # TOUTES les couches Dense
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch
    )

    print(f"Nombre de parametres (toutes couches): {influence_model.nb_params}")

    # KFAC
    kfac_ihvp = KFACIHVP(model=influence_model, train_dataset=dataset)

    print(f"\nNombre de couches KFAC: {len(kfac_ihvp.kfac_blocks)}")
    for idx, (A, G) in kfac_ihvp.kfac_blocks.items():
        info = kfac_ihvp.layer_info[idx]
        print(f"  Layer {idx} ({info['layer'].name}): A={A.shape}, G={G.shape}, n_params={info['n_params']}")

    # Tester la derniere couche (output)
    layer_idx = 1  # output layer
    A, G = kfac_ihvp.kfac_blocks[layer_idx]
    layer_info = kfac_ihvp.layer_info[layer_idx]

    print(f"\nTest sur la couche output (layer_idx={layer_idx}):")
    print(f"  A shape: {A.shape} (activation cov)")
    print(f"  G shape: {G.shape} (gradient cov)")

    # Prendre un vecteur v (gradient complet)
    test_batch = next(iter(dataset))
    v_full = influence_model.batch_jacobian_tensor(test_batch)[0]  # Premier sample
    print(f"\nVecteur v_full shape: {v_full.shape}")

    # Extraire le bloc pour la couche output
    param_start = kfac_ihvp.layer_info[0]['n_params']  # skip hidden layer
    n_params_output = layer_info['n_params']
    v_block = v_full[param_start:param_start + n_params_output].numpy()
    print(f"Vecteur v_block (output) shape: {v_block.shape}")

    # IHVP avec KFAC (votre implementation) - extraire le bloc output
    ihvp_full = kfac_ihvp._compute_ihvp_single_batch(test_batch)[:, 0].numpy()  # (nb_params,)
    ihvp_kfac_block = ihvp_full[param_start:param_start + n_params_output]
    print(f"IHVP KFAC block shape: {ihvp_kfac_block.shape}")

    # IHVP manuel selon la formule de reference (nngeometry/kronfluence)
    A_np = A.numpy()
    G_np = G.numpy()

    # Calculer pi-damping
    damping = kfac_ihvp.regul
    pi = np.sqrt((np.trace(A_np) / np.trace(G_np)) * (G_np.shape[0] / A_np.shape[0]))
    print(f"\nDamping: {damping}, Pi: {pi:.4f}")

    # Regulariser
    sqrt_damp = np.sqrt(damping)
    A_reg = A_np + sqrt_damp * pi * np.eye(A_np.shape[0])
    G_reg = G_np + sqrt_damp / pi * np.eye(G_np.shape[0])

    # V reshape
    out_dim = layer_info['out_features']
    in_dim = layer_info['in_features'] + (1 if layer_info['has_bias'] else 0)

    print(f"\nDimensions: in_dim={in_dim}, out_dim={out_dim}, n_params={n_params_output}")
    print(f"Expected: in_dim * out_dim = {in_dim * out_dim}")

    # Votre code utilise V de shape (in_dim, out_dim)
    V_in_out = v_block.reshape(in_dim, out_dim)
    V_out_in = v_block.reshape(out_dim, in_dim)

    print(f"V reshape (in, out): {V_in_out.shape}")
    print(f"V reshape (out, in): {V_out_in.shape}")

    # Formule nngeometry: solve(G, V) puis solve(A, X.T).T
    # avec V de shape (out, in)
    X1 = np.linalg.lstsq(G_reg, V_out_in, rcond=None)[0]
    result_out_in = np.linalg.lstsq(A_reg, X1.T, rcond=None)[0].T

    # Formule alternative avec V de shape (in, out)
    X2 = np.linalg.lstsq(G_reg, V_in_out.T, rcond=None)[0]
    result_in_out = np.linalg.lstsq(A_reg, X2.T, rcond=None)[0]

    ihvp_manual_out_in = result_out_in.flatten()
    ihvp_manual_in_out = result_in_out.flatten()

    print(f"\nComparaison (5 premiers elements):")
    print(f"  KFAC (votre code): {ihvp_kfac_block[:5]}...")
    print(f"  Manuel (out,in):   {ihvp_manual_out_in[:5]}...")
    print(f"  Manuel (in,out):   {ihvp_manual_in_out[:5]}...")

    # Cosine similarity
    cos_sim_out_in = np.dot(ihvp_kfac_block, ihvp_manual_out_in) / (
        np.linalg.norm(ihvp_kfac_block) * np.linalg.norm(ihvp_manual_out_in) + 1e-10)
    cos_sim_in_out = np.dot(ihvp_kfac_block, ihvp_manual_in_out) / (
        np.linalg.norm(ihvp_kfac_block) * np.linalg.norm(ihvp_manual_in_out) + 1e-10)

    print(f"\n  Cos sim (out,in): {cos_sim_out_in:.6f}")
    print(f"  Cos sim (in,out): {cos_sim_in_out:.6f}")

    if abs(cos_sim_out_in) > 0.99:
        print("  [PASS] Votre KFAC utilise la convention (out, in) comme nngeometry")
    elif abs(cos_sim_in_out) > 0.99:
        print("  [PASS] Votre KFAC utilise la convention (in, out)")
    else:
        print("  [WARN] Difference significative - verifier la formule")

    return ihvp_kfac_block, ihvp_manual_out_in, ihvp_manual_in_out


def test_layout_comparison():
    """Test 4: Verifie le layout des parametres (in_dim, out_dim) vs (out_dim, in_dim)."""
    print("\n" + "="*60)
    print("TEST 4: Layout des parametres")
    print("="*60)

    model = create_simple_mlp(input_dim=10, hidden_dim=8, output_dim=3)

    # Keras stocke Dense weights comme (in_features, out_features)
    hidden_layer = model.get_layer("hidden")
    output_layer = model.get_layer("output")

    print("Layout Keras (Dense):")
    print(f"  Hidden layer weight shape: {hidden_layer.kernel.shape}")  # (10, 8)
    print(f"  Hidden layer bias shape: {hidden_layer.bias.shape}")      # (8,)
    print(f"  Output layer weight shape: {output_layer.kernel.shape}")  # (8, 3)
    print(f"  Output layer bias shape: {output_layer.bias.shape}")      # (3,)

    print("\nPour KFAC:")
    print("  A devrait etre de taille (in_dim+1, in_dim+1) si bias")
    print("  G devrait etre de taille (out_dim, out_dim)")
    print(f"  Output: A expected (9, 9), G expected (3, 3)")

    # Verifier avec KFAC
    dataset, _, _ = create_test_data(n_samples=32, input_dim=10, batch_size=8)
    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    influence_model = InfluenceModel(
        model, start_layer=1, last_layer=-1,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch
    )

    kfac_ihvp = KFACIHVP(model=influence_model, train_dataset=dataset)

    for layer_idx, (A, G) in kfac_ihvp.kfac_blocks.items():
        layer_info = kfac_ihvp.layer_info[layer_idx]
        print(f"\nLayer {layer_idx} ({layer_info['layer'].name}):")
        print(f"  A shape: {A.shape}")
        print(f"  G shape: {G.shape}")
        print(f"  in_features: {layer_info['in_features']}, out_features: {layer_info['out_features']}")
        print(f"  has_bias: {layer_info['has_bias']}")


def test_influence_scores():
    """Test 5: Verifie que les scores d'influence sont coherents."""
    print("\n" + "="*60)
    print("TEST 5: Scores d'influence")
    print("="*60)

    model = create_simple_mlp(input_dim=8, hidden_dim=6, output_dim=3)
    dataset, X, y = create_test_data(n_samples=64, input_dim=8, n_classes=3, batch_size=8)

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    influence_model = InfluenceModel(
        model, start_layer=1, last_layer=-1,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch
    )

    # KFAC
    kfac_ihvp = KFACIHVP(model=influence_model, train_dataset=dataset)

    # Calculateur d'influence
    influence_calc = FirstOrderInfluenceCalculator(
        influence_model, dataset, kfac_ihvp
    )

    # Test samples
    test_dataset = tf.data.Dataset.from_tensor_slices((X[:4], y[:4])).batch(2)

    print("Calcul des scores d'influence...")

    # Calculer self-influence (devrait etre positive)
    for test_batch in test_dataset:
        influence_vector = influence_calc._compute_influence_vector(test_batch)
        print(f"  Influence vector shape: {influence_vector.shape}")
        print(f"  Influence vector min: {tf.reduce_min(influence_vector).numpy():.6e}")
        print(f"  Influence vector max: {tf.reduce_max(influence_vector).numpy():.6e}")
        print(f"  Has NaN: {tf.reduce_any(tf.math.is_nan(influence_vector)).numpy()}")
        break

    print("\n[INFO] Les scores d'influence devraient etre:")
    print("  - Positifs pour les samples qui 'aident' la prediction")
    print("  - Negatifs pour les samples qui 'nuisent'")
    print("  - Grands en valeur absolue pour les samples influents")


# =============================================================================
# TESTS EKFAC
# =============================================================================

def test_ekfac_eigendecomposition():
    """Test 6: Verifie l'eigendecomposition des facteurs KFAC pour EKFAC."""
    print("\n" + "="*60)
    print("TEST 6: EKFAC - Eigendecomposition des facteurs")
    print("="*60)

    model = create_simple_mlp(input_dim=8, hidden_dim=6, output_dim=3)
    dataset, X, y = create_test_data(n_samples=64, input_dim=8, n_classes=3, batch_size=16)

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    influence_model = InfluenceModel(
        model, start_layer=1, last_layer=-1,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch
    )

    print(f"Nombre de parametres: {influence_model.nb_params}")

    # EKFAC IHVP
    print("Calcul de EKFACIHVP...")
    ekfac_ihvp = EKFACIHVP(model=influence_model, train_dataset=dataset, update_eigen=True)

    print(f"\nNombre de couches EKFAC: {len(ekfac_ihvp.kfac_blocks)}")

    for layer_idx, (A, G) in ekfac_ihvp.kfac_blocks.items():
        info = ekfac_ihvp.layer_info[layer_idx]
        evecs_A, evecs_G = ekfac_ihvp.evecs[layer_idx]
        evals = ekfac_ihvp.evals[layer_idx]

        print(f"\nLayer {layer_idx} ({info['layer'].name}):")
        print(f"  A shape: {A.shape}, G shape: {G.shape}")
        print(f"  evecs_A shape: {evecs_A.shape}, evecs_G shape: {evecs_G.shape}")
        print(f"  evals shape: {evals.shape}")

        # Verifier que U_A @ diag(lambda_A) @ U_A^T ≈ A
        A_np = A.numpy()
        evecs_A_np = evecs_A.numpy()
        evals_A_manual, evecs_A_manual = np.linalg.eigh(A_np)

        # Reconstruction
        A_reconstructed = evecs_A_np @ np.diag(evals_A_manual) @ evecs_A_np.T

        reconstruction_error = np.linalg.norm(A_np - A_reconstructed) / np.linalg.norm(A_np)
        print(f"  A reconstruction error: {reconstruction_error:.6e}")

        # Meme chose pour G
        G_np = G.numpy()
        evecs_G_np = evecs_G.numpy()
        evals_G_manual, _ = np.linalg.eigh(G_np)

        G_reconstructed = evecs_G_np @ np.diag(evals_G_manual) @ evecs_G_np.T
        reconstruction_error_G = np.linalg.norm(G_np - G_reconstructed) / np.linalg.norm(G_np)
        print(f"  G reconstruction error: {reconstruction_error_G:.6e}")

        # Verifier les valeurs propres
        evals_np = evals.numpy()
        print(f"  Eigenvalues range: [{evals_np.min():.6e}, {evals_np.max():.6e}]")
        print(f"  Eigenvalues mean: {evals_np.mean():.6e}")

        if reconstruction_error < 1e-5 and reconstruction_error_G < 1e-5:
            print("  [PASS] Eigendecomposition correcte!")
        else:
            print("  [WARN] Erreur de reconstruction elevee")

    return ekfac_ihvp


def test_ekfac_vs_exact():
    """Test 7: Compare EKFAC IHVP avec ExactIHVP (reference)."""
    print("\n" + "="*60)
    print("TEST 7: EKFAC vs ExactIHVP")
    print("="*60)

    model = create_simple_mlp(input_dim=8, hidden_dim=6, output_dim=3)
    dataset, X, y = create_test_data(n_samples=32, input_dim=8, n_classes=3, batch_size=8)

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    influence_model = InfluenceModel(
        model, start_layer=1, last_layer=-1,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch
    )

    print(f"Nombre de parametres: {influence_model.nb_params}")

    # KFAC, EKFAC et ExactIHVP
    print("Calcul de KFACIHVP...")
    kfac_ihvp = KFACIHVP(model=influence_model, train_dataset=dataset)

    print("Calcul de EKFACIHVP (avec update_eigen=True)...")
    ekfac_ihvp = EKFACIHVP(model=influence_model, train_dataset=dataset, update_eigen=True)

    print("Calcul de ExactIHVP...")
    exact_ihvp = ExactIHVP(model=influence_model, train_dataset=dataset)

    # Comparer sur un batch de test
    test_dataset = tf.data.Dataset.from_tensor_slices((X[:4], y[:4])).batch(4)

    print("\nComparaison des IHVP:")
    for batch in test_dataset:
        # IHVP avec chaque methode
        exact_result = exact_ihvp._compute_ihvp_single_batch(batch).numpy()
        kfac_result = kfac_ihvp._compute_ihvp_single_batch(batch).numpy()
        ekfac_result = ekfac_ihvp._compute_ihvp_single_batch(batch).numpy()

        print(f"  ExactIHVP shape: {exact_result.shape}")
        print(f"  KFAC IHVP shape: {kfac_result.shape}")
        print(f"  EKFAC IHVP shape: {ekfac_result.shape}")

        # Calculer les cosine similarities
        batch_size = exact_result.shape[1]
        print("\n  Sample | Exact-KFAC | Exact-EKFAC | KFAC-EKFAC")
        print("  " + "-"*50)

        for i in range(min(4, batch_size)):
            exact_vec = exact_result[:, i].flatten()
            kfac_vec = kfac_result[:, i].flatten()
            ekfac_vec = ekfac_result[:, i].flatten()

            # Normaliser
            exact_norm = exact_vec / (np.linalg.norm(exact_vec) + 1e-10)
            kfac_norm = kfac_vec / (np.linalg.norm(kfac_vec) + 1e-10)
            ekfac_norm = ekfac_vec / (np.linalg.norm(ekfac_vec) + 1e-10)

            cos_exact_kfac = np.dot(exact_norm, kfac_norm)
            cos_exact_ekfac = np.dot(exact_norm, ekfac_norm)
            cos_kfac_ekfac = np.dot(kfac_norm, ekfac_norm)

            print(f"  {i:6d} | {cos_exact_kfac:10.4f} | {cos_exact_ekfac:11.4f} | {cos_kfac_ekfac:10.4f}")

        break

    print("\n[INFO] EKFAC devrait etre plus proche de Exact que KFAC")
    print("       car il corrige les valeurs propres approximatives de KFAC")

    return ekfac_ihvp, kfac_ihvp, exact_ihvp


def test_ekfac_vs_nngeometry():
    """Test 8: Compare EKFAC avec nngeometry PMatEKFAC."""
    print("\n" + "="*60)
    print("TEST 8: EKFAC vs nngeometry PMatEKFAC")
    print("="*60)

    # Setup: memes donnees pour TF et PyTorch
    np.random.seed(42)
    n_samples = 64
    input_dim = 8
    hidden_dim = 6
    output_dim = 3
    batch_size = 16

    X_np = np.random.randn(n_samples, input_dim).astype(np.float32)
    y_np = np.random.randint(0, output_dim, size=(n_samples,)).astype(np.int64)

    # === TensorFlow model ===
    tf_model = create_simple_mlp(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Get TF weights
    tf_hidden_w = tf_model.get_layer("hidden").kernel.numpy()
    tf_hidden_b = tf_model.get_layer("hidden").bias.numpy()
    tf_output_w = tf_model.get_layer("output").kernel.numpy()
    tf_output_b = tf_model.get_layer("output").bias.numpy()

    # === PyTorch model with SAME weights ===
    class PyTorchMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            return self.output(x)

    pt_model = PyTorchMLP()

    # Transfer weights: Keras (in, out) -> PyTorch (out, in)
    pt_model.hidden.weight.data = torch.tensor(tf_hidden_w.T)
    pt_model.hidden.bias.data = torch.tensor(tf_hidden_b)
    pt_model.output.weight.data = torch.tensor(tf_output_w.T)
    pt_model.output.bias.data = torch.tensor(tf_output_b)

    # Verify same forward pass
    x_test = torch.tensor(X_np[:2])
    tf_out = tf_model(X_np[:2]).numpy()
    pt_out = pt_model(x_test).detach().numpy()
    print(f"Forward pass match: {np.allclose(tf_out, pt_out, atol=1e-5)}")

    # === Compute EKFAC with TensorFlow/influenciae ===
    tf_dataset = tf.data.Dataset.from_tensor_slices((X_np, y_np.astype(np.int32))).batch(batch_size)
    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    influence_model = InfluenceModel(
        tf_model, start_layer=1, last_layer=-1,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch
    )
    ekfac_tf = EKFACIHVP(model=influence_model, train_dataset=tf_dataset, update_eigen=True)

    print(f"\nTF EKFAC computed. Layers: {len(ekfac_tf.kfac_blocks)}")
    for idx, (A, G) in ekfac_tf.kfac_blocks.items():
        info = ekfac_tf.layer_info[idx]
        evals = ekfac_tf.evals[idx]
        print(f"  Layer {idx} ({info['layer'].name}): A={A.shape}, G={G.shape}, evals={evals.shape}")

    # === Compute EKFAC with nngeometry ===
    pt_loader = DataLoader(TensorDataset(torch.tensor(X_np), torch.tensor(y_np)), batch_size=batch_size)

    F_ekfac = FIM(
        model=pt_model,
        loader=pt_loader,
        representation=PMatEKFAC,
        variant='classif_logits',
        device='cpu'
    )

    # Update eigenvalues (diag) with true values
    F_ekfac.update_diag(pt_loader)

    print("\nnngeometry EKFAC computed.")
    evecs_nn, diags_nn = F_ekfac.data
    for layer_id, diag in diags_nn.items():
        evecs_a, evecs_g = evecs_nn[layer_id]
        print(f"  Layer {layer_id}: evecs_a={evecs_a.shape}, evecs_g={evecs_g.shape}, diag={diag.shape}")

    # === Compare eigenvectors ===
    print("\n--- Comparaison des vecteurs propres ---")

    nn_layer_ids = list(diags_nn.keys())
    output_layer_id = nn_layer_ids[-1]

    # TF eigenvectors
    evecs_A_tf, evecs_G_tf = ekfac_tf.evecs[1]  # output layer
    evecs_A_tf = evecs_A_tf.numpy()
    evecs_G_tf = evecs_G_tf.numpy()

    # nngeometry eigenvectors
    evecs_a_nn, evecs_g_nn = evecs_nn[output_layer_id]
    evecs_a_nn = evecs_a_nn.numpy()
    evecs_g_nn = evecs_g_nn.numpy()

    print("\nOutput layer eigenvectors:")
    print(f"  TF evecs_A: {evecs_A_tf.shape}, nngeometry evecs_a: {evecs_a_nn.shape}")
    print(f"  TF evecs_G: {evecs_G_tf.shape}, nngeometry evecs_g: {evecs_g_nn.shape}")

    # Compare: eigenvectors can differ by sign, so compare |U^T @ V|
    # Should be close to identity if eigenvectors match
    similarity_A = np.abs(evecs_A_tf.T @ evecs_a_nn)
    similarity_G = np.abs(evecs_G_tf.T @ evecs_g_nn)

    # Check if diagonal is close to 1 (matching eigenvectors)
    diag_sim_A = np.mean(np.diag(similarity_A))
    diag_sim_G = np.mean(np.diag(similarity_G))

    print(f"  evecs_A diagonal similarity: {diag_sim_A:.6f}")
    print(f"  evecs_G diagonal similarity: {diag_sim_G:.6f}")

    # === Compare eigenvalues (diags) ===
    print("\n--- Comparaison des valeurs propres corrigees ---")

    evals_tf = ekfac_tf.evals[1].numpy()  # (out, in)
    diag_nn = diags_nn[output_layer_id].numpy()  # flattened (out * in,)

    print(f"  TF evals shape: {evals_tf.shape}")
    print(f"  nngeometry diag shape: {diag_nn.shape}")

    # Flatten TF evals to match nngeometry format
    # Note: nngeometry flattens as (out, in) row-major
    evals_tf_flat = evals_tf.flatten()

    # Normalize and compare
    evals_tf_norm = evals_tf_flat / (np.linalg.norm(evals_tf_flat) + 1e-10)
    diag_nn_norm = diag_nn / (np.linalg.norm(diag_nn) + 1e-10)

    cos_sim_evals = np.sum(evals_tf_norm * diag_nn_norm)
    print(f"  Eigenvalues cosine similarity: {cos_sim_evals:.6f}")

    # Compare ranges
    print(f"\n  TF evals: min={evals_tf.min():.6e}, max={evals_tf.max():.6e}, mean={evals_tf.mean():.6e}")
    print(f"  NN diag:  min={diag_nn.min():.6e}, max={diag_nn.max():.6e}, mean={diag_nn.mean():.6e}")

    # === Compare solve ===
    print("\n--- Comparaison du solve EKFAC ---")

    # Create a gradient vector with PyTorch
    x_sample = torch.tensor(X_np[:1], requires_grad=True)
    y_sample = torch.tensor(y_np[:1])

    pt_model.zero_grad()
    logits = pt_model(x_sample)
    loss = nn.CrossEntropyLoss()(logits, y_sample)
    loss.backward()

    # Get gradient as PVector for nngeometry
    layer_collection = LayerCollection.from_model(pt_model)
    grad_dict = {}
    for name, param in pt_model.named_parameters():
        layer_name = name.split('.')[0]
        param_type = name.split('.')[1]
        if layer_name not in grad_dict:
            grad_dict[layer_name] = [None, None]
        if param_type == 'weight':
            grad_dict[layer_name][0] = param.grad.clone()
        else:
            grad_dict[layer_name][1] = param.grad.clone()

    grad_pvector_dict = {}
    for layer_id in diags_nn.keys():
        w_grad = grad_dict[layer_id][0]
        b_grad = grad_dict[layer_id][1]
        grad_pvector_dict[layer_id] = (w_grad, b_grad)

    grad_pvector = PVector(layer_collection=layer_collection, dict_repr=grad_pvector_dict)

    # Solve with nngeometry EKFAC
    damping = ekfac_tf.regul
    solved_nn = F_ekfac.solvePVec(grad_pvector, regul=damping)

    # Get the output layer result
    solved_nn_output = solved_nn.to_dict()[output_layer_id]
    solved_nn_w = solved_nn_output[0].numpy().flatten()
    solved_nn_b = solved_nn_output[1].numpy().flatten()
    solved_nn_vec = np.concatenate([solved_nn_w, solved_nn_b])

    print(f"nngeometry EKFAC solve output shape: {solved_nn_vec.shape}")

    # Solve with TF EKFAC
    tf_batch = (tf.constant(X_np[:1]), tf.constant(y_np[:1].astype(np.int32)))
    ihvp_tf = ekfac_tf._compute_ihvp_single_batch(tf_batch)[:, 0].numpy()

    # Extract output layer block
    param_start = ekfac_tf.layer_info[0]['n_params']
    n_params_output = ekfac_tf.layer_info[1]['n_params']
    ihvp_tf_output = ihvp_tf[param_start:param_start + n_params_output]

    print(f"TF EKFAC solve output shape: {ihvp_tf_output.shape}")

    # Compare
    cos_sim_solve = np.dot(ihvp_tf_output, solved_nn_vec) / (
        np.linalg.norm(ihvp_tf_output) * np.linalg.norm(solved_nn_vec) + 1e-10)

    print("\nSolve comparison:")
    print(f"  TF first 5: {ihvp_tf_output[:5]}")
    print(f"  NN first 5: {solved_nn_vec[:5]}")
    print(f"  Cosine similarity: {cos_sim_solve:.6f}")

    if abs(cos_sim_solve) > 0.95:
        print("  [PASS] TF EKFAC matches nngeometry")
    else:
        print("  [WARN] Difference between TF EKFAC and nngeometry")
        print("  Note: May be due to different ordering or eigenvalue computation")

    return ekfac_tf, F_ekfac


def test_ekfac_solve_formula():
    """Test 9: Verifie la formule de solve EKFAC."""
    print("\n" + "="*60)
    print("TEST 9: Formule EKFAC solve")
    print("="*60)

    model = create_simple_mlp(input_dim=6, hidden_dim=4, output_dim=3)
    dataset, X, y = create_test_data(n_samples=32, input_dim=6, n_classes=3, batch_size=8)

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    influence_model = InfluenceModel(
        model, start_layer=1, last_layer=-1,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch
    )

    print(f"Nombre de parametres: {influence_model.nb_params}")

    # EKFAC
    ekfac_ihvp = EKFACIHVP(model=influence_model, train_dataset=dataset, update_eigen=True)

    print(f"\nNombre de couches EKFAC: {len(ekfac_ihvp.kfac_blocks)}")

    # Tester sur la couche output
    layer_idx = 1
    layer_info = ekfac_ihvp.layer_info[layer_idx]
    evecs_A, evecs_G = ekfac_ihvp.evecs[layer_idx]
    evals = ekfac_ihvp.evals[layer_idx]

    print(f"\nTest sur la couche output (layer_idx={layer_idx}):")
    print(f"  evecs_A shape: {evecs_A.shape}")
    print(f"  evecs_G shape: {evecs_G.shape}")
    print(f"  evals shape: {evals.shape}")

    # Prendre un vecteur v
    test_batch = next(iter(dataset))
    v_full = influence_model.batch_jacobian_tensor(test_batch)[0]

    # Extraire le bloc output
    param_start = ekfac_ihvp.layer_info[0]['n_params']
    n_params_output = layer_info['n_params']
    v_block = v_full[param_start:param_start + n_params_output].numpy()

    print(f"\nVecteur v_block shape: {v_block.shape}")

    # IHVP avec EKFAC
    ihvp_full = ekfac_ihvp._compute_ihvp_single_batch(test_batch)[:, 0].numpy()
    ihvp_ekfac_block = ihvp_full[param_start:param_start + n_params_output]

    print(f"IHVP EKFAC block shape: {ihvp_ekfac_block.shape}")

    # IHVP manuel selon la formule EKFAC
    # Dans l'eigenbasis: solve = v_kfe / (evals + damping)
    # v_kfe = U_G^T @ V @ U_A (ou V est reshape)
    # result = U_G @ solve_kfe @ U_A^T

    evecs_A_np = evecs_A.numpy()
    evecs_G_np = evecs_G.numpy()
    evals_np = evals.numpy()
    damping = ekfac_ihvp.regul

    out_dim = layer_info['out_features']
    in_dim = layer_info['in_features'] + (1 if layer_info['has_bias'] else 0)

    print(f"\nDimensions: in_dim={in_dim}, out_dim={out_dim}")

    # V reshape: TF utilise (in, out)
    V = v_block.reshape(in_dim, out_dim)

    # Projet vers eigenbasis: V_kfe = U_A^T @ V @ U_G
    V_kfe = evecs_A_np.T @ V @ evecs_G_np  # (in, out)

    # Solve dans l'eigenbasis: division element-wise
    # Note: evals est (out, in), V_kfe est (in, out)
    evals_T = evals_np.T  # (in, out)
    solve_kfe = V_kfe / (evals_T + damping)

    # Handle numerical issues
    solve_kfe = np.where(np.isfinite(solve_kfe), solve_kfe, 0)

    # Projet retour: X = U_A @ solve_kfe @ U_G^T
    X = evecs_A_np @ solve_kfe @ evecs_G_np.T  # (in, out)

    ihvp_manual = X.flatten()

    print("\nComparaison (5 premiers elements):")
    print(f"  EKFAC (votre code): {ihvp_ekfac_block[:5]}...")
    print(f"  Manuel:             {ihvp_manual[:5]}...")

    # Cosine similarity
    cos_sim = np.dot(ihvp_ekfac_block, ihvp_manual) / (
        np.linalg.norm(ihvp_ekfac_block) * np.linalg.norm(ihvp_manual) + 1e-10)

    print(f"\n  Cosine similarity: {cos_sim:.6f}")

    if abs(cos_sim) > 0.99:
        print("  [PASS] Formule EKFAC correcte!")
    else:
        print("  [WARN] Difference significative - verifier la formule")

    return ihvp_ekfac_block, ihvp_manual


def test_ekfac_eigenvalue_update():
    """Test 10: Verifie la mise a jour des valeurs propres EKFAC."""
    print("\n" + "="*60)
    print("TEST 10: EKFAC - Mise a jour des valeurs propres")
    print("="*60)

    model = create_simple_mlp(input_dim=8, hidden_dim=6, output_dim=3)
    dataset, X, y = create_test_data(n_samples=64, input_dim=8, n_classes=3, batch_size=16)

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    influence_model = InfluenceModel(
        model, start_layer=1, last_layer=-1,
        loss_function=loss_fn,
        process_batch_for_loss_fn=process_batch
    )

    # EKFAC sans correction (comme KFAC mais dans eigenbasis)
    print("Calcul de EKFACIHVP sans correction (update_eigen=False)...")
    ekfac_no_update = EKFACIHVP(model=influence_model, train_dataset=dataset, update_eigen=False)

    # EKFAC avec correction
    print("Calcul de EKFACIHVP avec correction (update_eigen=True)...")
    ekfac_with_update = EKFACIHVP(model=influence_model, train_dataset=dataset, update_eigen=True)

    print("\nComparaison des valeurs propres:")

    for layer_idx in ekfac_no_update.evals.keys():
        evals_no_update = ekfac_no_update.evals[layer_idx].numpy()
        evals_with_update = ekfac_with_update.evals[layer_idx].numpy()

        layer_info = ekfac_no_update.layer_info[layer_idx]

        print(f"\nLayer {layer_idx} ({layer_info['layer'].name}):")
        print(f"  Sans correction: min={evals_no_update.min():.6e}, max={evals_no_update.max():.6e}")
        print(f"  Avec correction: min={evals_with_update.min():.6e}, max={evals_with_update.max():.6e}")

        # Calculer la difference relative
        diff = np.abs(evals_with_update - evals_no_update)
        rel_diff = diff / (np.abs(evals_no_update) + 1e-10)

        print(f"  Difference relative moyenne: {rel_diff.mean():.4f}")
        print(f"  Difference relative max: {rel_diff.max():.4f}")

        # Comparer avec la formule theorique
        # Sans correction: evals = evals_G[:, None] * evals_A[None, :]
        # Avec correction: evals = (1/n) * sum_samples (g_kfe**2) outer (a_kfe**2)

        A, G = ekfac_no_update.kfac_blocks[layer_idx]
        evals_A_direct, _ = np.linalg.eigh(A.numpy())
        evals_G_direct, _ = np.linalg.eigh(G.numpy())

        # Produit externe (comme initialisation sans correction)
        expected_no_update = np.outer(evals_G_direct, evals_A_direct)

        cos_sim_no_update = np.sum(evals_no_update * expected_no_update) / (
            np.linalg.norm(evals_no_update) * np.linalg.norm(expected_no_update) + 1e-10)

        print(f"  Cos sim (sans correction vs produit externe): {cos_sim_no_update:.6f}")

    # Comparer les IHVP
    print("\n--- Comparaison des IHVP ---")

    test_batch = next(iter(dataset))
    ihvp_no_update = ekfac_no_update._compute_ihvp_single_batch(test_batch)[:, 0].numpy()
    ihvp_with_update = ekfac_with_update._compute_ihvp_single_batch(test_batch)[:, 0].numpy()

    cos_sim_ihvp = np.dot(ihvp_no_update, ihvp_with_update) / (
        np.linalg.norm(ihvp_no_update) * np.linalg.norm(ihvp_with_update) + 1e-10)

    print(f"Cosine similarity entre IHVP (sans/avec correction): {cos_sim_ihvp:.6f}")
    print(f"Norme IHVP sans correction: {np.linalg.norm(ihvp_no_update):.6f}")
    print(f"Norme IHVP avec correction: {np.linalg.norm(ihvp_with_update):.6f}")

    print("\n[INFO] Les valeurs propres corrigees devraient etre plus precises")
    print("       car elles prennent en compte les correlations entre A et G")

    return ekfac_no_update, ekfac_with_update


def test_influence_scores_comparison():
    """Test 11: Compare les scores d'influence entre influenciae et nngeometry.

    C'est le test le plus significatif car il compare les resultats finaux:
    influence(z_test, z_train) = -g_test^T @ H^{-1} @ g_train
    """
    print("\n" + "="*60)
    print("TEST 11: Comparaison des scores d'influence")
    print("="*60)

    # Configuration
    input_dim = 6
    hidden_dim = 4
    output_dim = 3
    n_train = 32
    n_test = 4
    batch_size = 8

    np.random.seed(42)
    torch.manual_seed(42)
    tf.random.set_seed(42)

    # Creer donnees
    X_train = np.random.randn(n_train, input_dim).astype(np.float32)
    y_train = np.random.randint(0, output_dim, n_train).astype(np.int64)

    X_test = np.random.randn(n_test, input_dim).astype(np.float32)
    y_test = np.random.randint(0, output_dim, n_test).astype(np.int64)

    # === Modele TensorFlow (API Functional pour compatibilite KFAC) ===
    inputs = tf.keras.layers.Input(shape=(input_dim,), name='input')
    x = tf.keras.layers.Dense(hidden_dim, activation='relu', name='hidden',
                              kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
    outputs = tf.keras.layers.Dense(output_dim, name='output',
                                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    tf_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='test_mlp')

    # === Modele PyTorch avec memes poids ===
    pt_model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    pt_model[0]._get_name = lambda: 'hidden'
    pt_model[2]._get_name = lambda: 'output'

    # Copier les poids TF -> PyTorch (layers[1] = hidden, layers[2] = output car layers[0] = Input)
    with torch.no_grad():
        pt_model[0].weight.data = torch.tensor(tf_model.layers[1].kernel.numpy().T)
        pt_model[0].bias.data = torch.tensor(tf_model.layers[1].bias.numpy())
        pt_model[2].weight.data = torch.tensor(tf_model.layers[2].kernel.numpy().T)
        pt_model[2].bias.data = torch.tensor(tf_model.layers[2].bias.numpy())

    # Verifier que les modeles donnent le meme output
    with torch.no_grad():
        pt_out = pt_model(torch.tensor(X_test)).numpy()
    tf_out = tf_model(X_test).numpy()
    print(f"Forward pass match: {np.allclose(pt_out, tf_out, atol=1e-5)}")

    # === influenciae: KFAC et EKFAC ===
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)

    # start_layer=1 saute l'Input layer, last_layer=-1 = output layer
    influence_model = InfluenceModel(
        tf_model, start_layer=1, last_layer=-1,
        loss_function=loss_fn
    )

    print("\nCalcul de KFACIHVP...")
    kfac_tf = KFACIHVP(model=influence_model, train_dataset=train_dataset)

    print("Calcul de EKFACIHVP...")
    ekfac_tf = EKFACIHVP(model=influence_model, train_dataset=train_dataset, update_eigen=True)

    # Damping pour nngeometry (leur API l'exige explicitement dans solve())
    damping_nn = 0.01

    # Calculer les gradients pour les samples test
    print("\nCalcul des gradients test...")
    test_grads_tf = []
    for i in range(n_test):
        with tf.GradientTape() as tape:
            logits = tf_model(X_test[i:i+1])
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_test[i:i+1], logits, from_logits=True)
        grads = tape.gradient(loss, tf_model.trainable_variables)
        grad_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        test_grads_tf.append(grad_flat.numpy())
    test_grads_tf = np.array(test_grads_tf)  # (n_test, n_params)

    # Calculer les gradients pour les samples train
    print("Calcul des gradients train...")
    train_grads_tf = []
    for i in range(n_train):
        with tf.GradientTape() as tape:
            logits = tf_model(X_train[i:i+1])
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_train[i:i+1], logits, from_logits=True)
        grads = tape.gradient(loss, tf_model.trainable_variables)
        grad_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        train_grads_tf.append(grad_flat.numpy())
    train_grads_tf = np.array(train_grads_tf)  # (n_train, n_params)

    # Calculer IHVP pour les gradients test
    print("Calcul des IHVP pour les samples test...")

    # KFAC IHVP
    ihvp_kfac = []
    for i in range(n_test):
        test_grad_tensor = tf.constant(test_grads_tf[i:i+1])
        ihvp = kfac_tf._compute_ihvp_single_batch((test_grad_tensor,), use_gradient=False)
        ihvp_kfac.append(ihvp.numpy().flatten())
    ihvp_kfac = np.array(ihvp_kfac)  # (n_test, n_params)

    # EKFAC IHVP
    ihvp_ekfac = []
    for i in range(n_test):
        test_grad_tensor = tf.constant(test_grads_tf[i:i+1])
        ihvp = ekfac_tf._compute_ihvp_single_batch((test_grad_tensor,), use_gradient=False)
        ihvp_ekfac.append(ihvp.numpy().flatten())
    ihvp_ekfac = np.array(ihvp_ekfac)  # (n_test, n_params)

    # Scores d'influence: influence[i,j] = -g_test[i]^T @ H^{-1} @ g_train[j]
    # = -ihvp_test[i]^T @ g_train[j]
    print("\nCalcul des scores d'influence TF...")
    influence_kfac = -ihvp_kfac @ train_grads_tf.T  # (n_test, n_train)
    influence_ekfac = -ihvp_ekfac @ train_grads_tf.T  # (n_test, n_train)

    # === nngeometry ===
    print("\nCalcul avec nngeometry...")

    # Dataset PyTorch
    X_train_pt = torch.tensor(X_train)
    y_train_pt = torch.tensor(y_train)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_pt, y_train_pt),
        batch_size=batch_size
    )

    # KFAC nngeometry (utiliser FIM comme dans les autres tests)
    kfac_nn = FIM(
        model=pt_model,
        loader=train_loader,
        representation=PMatKFAC,
        variant='classif_logits',
        device='cpu'
    )

    # EKFAC nngeometry
    ekfac_nn = FIM(
        model=pt_model,
        loader=train_loader,
        representation=PMatEKFAC,
        variant='classif_logits',
        device='cpu'
    )
    # Mise a jour des eigenvalues diagonales pour EKFAC (comme dans le code de reference)
    ekfac_nn.update_diag(examples=train_loader)

    # Calculer les gradients PyTorch pour test samples
    print("Calcul des gradients test (PyTorch)...")
    test_grads_pt = []
    for i in range(n_test):
        pt_model.zero_grad()
        x = torch.tensor(X_test[i:i+1], requires_grad=False)
        y = torch.tensor(y_test[i:i+1])
        logits = pt_model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        grad_flat = torch.cat([p.grad.flatten() for p in pt_model.parameters()])
        test_grads_pt.append(grad_flat.numpy())
    test_grads_pt = np.array(test_grads_pt)

    # Calculer les gradients PyTorch pour train samples
    print("Calcul des gradients train (PyTorch)...")
    train_grads_pt = []
    for i in range(n_train):
        pt_model.zero_grad()
        x = torch.tensor(X_train[i:i+1], requires_grad=False)
        y = torch.tensor(y_train[i:i+1])
        logits = pt_model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        grad_flat = torch.cat([p.grad.flatten() for p in pt_model.parameters()])
        train_grads_pt.append(grad_flat.numpy())
    train_grads_pt = np.array(train_grads_pt)

    # IHVP avec nngeometry
    print("Calcul des IHVP (nngeometry)...")

    # Creer LayerCollection pour PVector
    layer_collection = LayerCollection.from_model(pt_model)

    ihvp_kfac_nn = []
    ihvp_ekfac_nn = []
    for i in range(n_test):
        # Calculer le gradient test
        pt_model.zero_grad()
        x = torch.tensor(X_test[i:i+1], requires_grad=False)
        y = torch.tensor(y_test[i:i+1])
        logits = pt_model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        # Construire PVector a partir des gradients
        grad_dict = {}
        for name, param in pt_model.named_parameters():
            layer_name = name.split('.')[0]
            param_type = name.split('.')[1]
            if layer_name not in grad_dict:
                grad_dict[layer_name] = [None, None]
            if param_type == 'weight':
                grad_dict[layer_name][0] = param.grad.clone()
            else:
                grad_dict[layer_name][1] = param.grad.clone()

        grad_pvector_dict = {}
        for layer_id in kfac_nn.data.keys():
            grad_pvector_dict[layer_id] = (grad_dict[layer_id][0], grad_dict[layer_id][1])

        grad_pvec = PVector(layer_collection=layer_collection, dict_repr=grad_pvector_dict)

        # Solve avec KFAC (utiliser F.solve comme dans le code de reference)
        ihvp_kfac_pvec = kfac_nn.solve(grad_pvec, regul=damping_nn)
        # Convertir PVector en tensor puis numpy
        ihvp_kfac_tensor = ihvp_kfac_pvec.to_torch()
        ihvp_kfac_nn.append(ihvp_kfac_tensor.numpy())

        # Solve avec EKFAC
        ihvp_ekfac_pvec = ekfac_nn.solve(grad_pvec, regul=damping_nn)
        ihvp_ekfac_tensor = ihvp_ekfac_pvec.to_torch()
        ihvp_ekfac_nn.append(ihvp_ekfac_tensor.numpy())

    ihvp_kfac_nn = np.array(ihvp_kfac_nn)
    ihvp_ekfac_nn = np.array(ihvp_ekfac_nn)

    # Scores d'influence nngeometry
    print("Calcul des scores d'influence nngeometry...")
    influence_kfac_nn = -ihvp_kfac_nn @ train_grads_pt.T
    influence_ekfac_nn = -ihvp_ekfac_nn @ train_grads_pt.T

    # === Comparaison ===
    print("\n" + "="*60)
    print("COMPARAISON DES SCORES D'INFLUENCE")
    print("="*60)

    print(f"\nShape des scores: ({n_test}, {n_train})")
    print(f"  influence[i,j] = influence du train sample j sur test sample i")

    # Afficher les scores pour le premier test sample
    print(f"\n--- Scores pour test sample 0 (premiers 10 train samples) ---")
    print(f"  KFAC TF:      {influence_kfac[0, :10]}")
    print(f"  KFAC nngeom:  {influence_kfac_nn[0, :10]}")
    print(f"  EKFAC TF:     {influence_ekfac[0, :10]}")
    print(f"  EKFAC nngeom: {influence_ekfac_nn[0, :10]}")

    # Correlation/cosine similarity par test sample
    print(f"\n--- Correlation des scores par test sample ---")
    print(f"  Test | KFAC TF-NN | EKFAC TF-NN | KFAC-EKFAC TF | KFAC-EKFAC NN")
    print(f"  " + "-"*65)

    for i in range(n_test):
        # KFAC: TF vs nngeometry
        cos_kfac = np.corrcoef(influence_kfac[i], influence_kfac_nn[i])[0, 1]

        # EKFAC: TF vs nngeometry
        cos_ekfac = np.corrcoef(influence_ekfac[i], influence_ekfac_nn[i])[0, 1]

        # KFAC vs EKFAC (TF)
        cos_kfac_ekfac_tf = np.corrcoef(influence_kfac[i], influence_ekfac[i])[0, 1]

        # KFAC vs EKFAC (nngeometry)
        cos_kfac_ekfac_nn = np.corrcoef(influence_kfac_nn[i], influence_ekfac_nn[i])[0, 1]

        print(f"    {i} |    {cos_kfac:+.4f} |     {cos_ekfac:+.4f} |       {cos_kfac_ekfac_tf:+.4f} |        {cos_kfac_ekfac_nn:+.4f}")

    # Correlation globale (tous les scores)
    print(f"\n--- Correlation globale (tous les scores flatten) ---")

    cos_kfac_global = np.corrcoef(influence_kfac.flatten(), influence_kfac_nn.flatten())[0, 1]
    cos_ekfac_global = np.corrcoef(influence_ekfac.flatten(), influence_ekfac_nn.flatten())[0, 1]
    cos_kfac_ekfac_tf_global = np.corrcoef(influence_kfac.flatten(), influence_ekfac.flatten())[0, 1]

    print(f"  KFAC TF vs nngeometry:  {cos_kfac_global:+.4f}")
    print(f"  EKFAC TF vs nngeometry: {cos_ekfac_global:+.4f}")
    print(f"  KFAC vs EKFAC (TF):     {cos_kfac_ekfac_tf_global:+.4f}")

    # Ranking comparison (Spearman)
    from scipy.stats import spearmanr

    print(f"\n--- Correlation de rang (Spearman) ---")
    print(f"  (Important pour l'interpretabilite: est-ce que les samples les plus influents sont les memes?)")

    for i in range(min(2, n_test)):  # Premiers 2 test samples
        spearman_kfac, _ = spearmanr(influence_kfac[i], influence_kfac_nn[i])
        spearman_ekfac, _ = spearmanr(influence_ekfac[i], influence_ekfac_nn[i])
        print(f"  Test {i}: KFAC Spearman={spearman_kfac:+.4f}, EKFAC Spearman={spearman_ekfac:+.4f}")

    # Top-k agreement
    print(f"\n--- Top-k Agreement ---")
    print(f"  (Est-ce que les k samples les plus influents sont les memes?)")

    for k in [3, 5]:
        agreements_kfac = []
        agreements_ekfac = []
        for i in range(n_test):
            top_k_tf = set(np.argsort(np.abs(influence_kfac[i]))[-k:])
            top_k_nn = set(np.argsort(np.abs(influence_kfac_nn[i]))[-k:])
            agreements_kfac.append(len(top_k_tf & top_k_nn) / k)

            top_k_tf = set(np.argsort(np.abs(influence_ekfac[i]))[-k:])
            top_k_nn = set(np.argsort(np.abs(influence_ekfac_nn[i]))[-k:])
            agreements_ekfac.append(len(top_k_tf & top_k_nn) / k)

        print(f"  Top-{k}: KFAC={np.mean(agreements_kfac):.2%}, EKFAC={np.mean(agreements_ekfac):.2%}")

    # Verdict
    print(f"\n--- Verdict ---")
    if cos_kfac_global > 0.8:
        print(f"  [PASS] KFAC: Bonne correspondance avec nngeometry (corr={cos_kfac_global:.4f})")
    elif cos_kfac_global > 0.5:
        print(f"  [WARN] KFAC: Correspondance moderee avec nngeometry (corr={cos_kfac_global:.4f})")
    else:
        print(f"  [FAIL] KFAC: Faible correspondance avec nngeometry (corr={cos_kfac_global:.4f})")

    if cos_ekfac_global > 0.8:
        print(f"  [PASS] EKFAC: Bonne correspondance avec nngeometry (corr={cos_ekfac_global:.4f})")
    elif cos_ekfac_global > 0.5:
        print(f"  [WARN] EKFAC: Correspondance moderee avec nngeometry (corr={cos_ekfac_global:.4f})")
    else:
        print(f"  [FAIL] EKFAC: Faible correspondance avec nngeometry (corr={cos_ekfac_global:.4f})")

    return {
        'influence_kfac_tf': influence_kfac,
        'influence_ekfac_tf': influence_ekfac,
        'influence_kfac_nn': influence_kfac_nn,
        'influence_ekfac_nn': influence_ekfac_nn,
    }


def run_all_tests():
    """Execute tous les tests de validation."""
    print("="*60)
    print("VALIDATION KFAC et EKFAC - influenciae vs nngeometry/kronfluence")
    print("="*60)

    # ==================== TESTS KFAC ====================
    print("\n" + "="*60)
    print("TESTS KFAC")
    print("="*60)

    try:
        test_kfac_factors()
    except Exception as e:
        print(f"[ERROR] Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_kfac_vs_nngeometry()
    except Exception as e:
        print(f"[ERROR] Test 3a (nngeometry) failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_kfac_solve_formula()
    except Exception as e:
        print(f"[ERROR] Test 3b failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_layout_comparison()
    except Exception as e:
        print(f"[ERROR] Test 4 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_influence_scores()
    except Exception as e:
        print(f"[ERROR] Test 5 failed: {e}")
        import traceback
        traceback.print_exc()

    # ==================== TESTS EKFAC ====================
    print("\n" + "="*60)
    print("TESTS EKFAC")
    print("="*60)

    try:
        test_ekfac_eigendecomposition()
    except Exception as e:
        print(f"[ERROR] Test 6 (EKFAC eigendecomposition) failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_ekfac_vs_nngeometry()
    except Exception as e:
        print(f"[ERROR] Test 8 (EKFAC vs nngeometry) failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_ekfac_solve_formula()
    except Exception as e:
        print(f"[ERROR] Test 9 (EKFAC solve formula) failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_ekfac_eigenvalue_update()
    except Exception as e:
        print(f"[ERROR] Test 10 (EKFAC eigenvalue update) failed: {e}")
        import traceback
        traceback.print_exc()

    # ==================== TEST INFLUENCE SCORES ====================
    print("\n" + "="*60)
    print("TEST COMPARAISON SCORES D'INFLUENCE")
    print("="*60)

    try:
        test_influence_scores_comparison()
    except Exception as e:
        print(f"[ERROR] Test 11 (Influence scores comparison) failed: {e}")
        import traceback
        traceback.print_exc()

    # ==================== TESTS LENTS ====================
    print("\n" + "="*60)
    print("TESTS LENTS (calcul Hessienne complete)")
    print("="*60)
    run_slow = input("Executer les tests lents? (y/n): ").strip().lower() == 'y'

    if run_slow:
        try:
            test_kfac_vs_exact()
        except Exception as e:
            print(f"[ERROR] Test 2 (KFAC vs Exact) failed: {e}")
            import traceback
            traceback.print_exc()

        try:
            test_hvp_ihvp_identity()
        except Exception as e:
            print(f"[ERROR] Test 3 (HVP/IHVP identity) failed: {e}")
            import traceback
            traceback.print_exc()

        try:
            test_ekfac_vs_exact()
        except Exception as e:
            print(f"[ERROR] Test 7 (EKFAC vs Exact) failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("VALIDATION TERMINEE")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
