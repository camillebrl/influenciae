"""
Token-wise Influence module implementing per-token influence attribution.

This computes the influence of each input token on the prediction by computing:
    influence_token[i] = d/d(embedding_token_i) [ihvp^T @ grad_theta L(x_train, y_train)]

Where:
- ihvp = H^{-1} @ grad_theta L(x_test, y_test) is computed using EKFAC/KFAC
- grad_theta L(x_train, y_train) is the gradient of training loss w.r.t. parameters
- We backprop through input embeddings to get per-token attributions

Based on the pixelwise influence approach from:
- "Understanding Black-box Predictions via Influence Functions" (Koh & Liang, 2017)
"""
import tensorflow as tf

from ..common import InfluenceModel
from ..common import InverseHessianVectorProduct, IHVPCalculator
from ..types import Optional, Union, Tuple


class TokenwiseInfluenceCalculator:
    """
    A class implementing token-wise influence attribution.

    For each training sample, computes how each input token/feature contributes
    to the influence on a test sample's prediction.

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface.
    ihvp_calculator
        An IHVPCalculator object (e.g., KFACIHVP, EKFACIHVP).
    """

    def __init__(
            self,
            model: InfluenceModel,
            ihvp_calculator: Union[InverseHessianVectorProduct, IHVPCalculator]
    ):
        self.model = model
        self.ihvp_calculator = ihvp_calculator

    def compute_tokenwise_influence(
            self,
            x_train: tf.Tensor,
            y_train: tf.Tensor,
            x_test: tf.Tensor,
            y_test: tf.Tensor,
            sample_weight_train: Optional[tf.Tensor] = None,
            sample_weight_test: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Compute the per-token influence of a training sample on a test sample.

        This implements the formula:
            influence_token = d/d(x_train) [ihvp^T @ grad_theta L(x_train, y_train)]

        Where ihvp = H^{-1} @ grad_theta L(x_test, y_test)

        Parameters
        ----------
        x_train
            Training input tensor of shape (1, seq_len, emb_dim) or (1, n_features)
        y_train
            Training label tensor of shape (1,) or (1, n_classes)
        x_test
            Test input tensor of shape (1, seq_len, emb_dim) or (1, n_features)
        y_test
            Test label tensor of shape (1,) or (1, n_classes)
        sample_weight_train
            Optional sample weight for training sample
        sample_weight_test
            Optional sample weight for test sample

        Returns
        -------
        influence_map
            Tensor of shape matching x_train, containing per-token/feature influence scores
        """
        # Ensure batch dimension
        if len(x_train.shape) == 1:
            x_train = tf.expand_dims(x_train, 0)
        if len(x_test.shape) == 1:
            x_test = tf.expand_dims(x_test, 0)
        if len(y_train.shape) == 0:
            y_train = tf.expand_dims(y_train, 0)
        if len(y_test.shape) == 0:
            y_test = tf.expand_dims(y_test, 0)

        # Step 1: Compute IHVP for the test sample
        # ihvp = H^{-1} @ grad_theta L(x_test, y_test)
        if sample_weight_test is not None:
            test_batch = (x_test, y_test, sample_weight_test)
        else:
            test_batch = (x_test, y_test)

        ihvp = self.ihvp_calculator._compute_ihvp_single_batch(test_batch)
        ihvp_vec = tf.reshape(ihvp, [-1])  # Flatten to (n_params,)

        # Step 2: Compute grad_theta L(x_train, y_train) with gradient tape watching x_train
        # We need a nested tape to backprop through the gradient computation
        x_train_var = tf.Variable(x_train, trainable=True)

        with tf.GradientTape() as outer_tape:
            outer_tape.watch(x_train_var)

            with tf.GradientTape() as inner_tape:
                # Forward pass
                predictions = self.model.model(x_train_var, training=True)

                # Compute loss
                if sample_weight_train is not None:
                    loss = self.model.loss_function(y_train, predictions, sample_weight_train)
                else:
                    loss = self.model.loss_function(y_train, predictions)

                # Handle per-sample loss (no reduction)
                if len(loss.shape) > 0:
                    loss = tf.reduce_sum(loss)

            # Compute gradients w.r.t. model parameters
            trainable_vars = self.model.weights
            grad_train = inner_tape.gradient(loss, trainable_vars)

            # Flatten gradients to match IHVP shape
            grad_train_vec = tf.concat([tf.reshape(g, [-1]) for g in grad_train], axis=0)

            # Dot product with IHVP
            influence_scalar = tf.reduce_sum(ihvp_vec * grad_train_vec)

        # Step 3: Backprop through input to get per-token influence
        influence_map = outer_tape.gradient(influence_scalar, x_train_var)

        return influence_map

    def compute_tokenwise_influence_batch(
            self,
            x_train_batch: tf.Tensor,
            y_train_batch: tf.Tensor,
            x_test: tf.Tensor,
            y_test: tf.Tensor,
            sample_weight_train: Optional[tf.Tensor] = None,
            sample_weight_test: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Compute per-token influence for a batch of training samples.

        Parameters
        ----------
        x_train_batch
            Training input batch of shape (batch_size, seq_len, emb_dim) or (batch_size, n_features)
        y_train_batch
            Training labels of shape (batch_size,) or (batch_size, n_classes)
        x_test
            Single test input of shape (1, seq_len, emb_dim) or (1, n_features)
        y_test
            Single test label of shape (1,) or (1, n_classes)
        sample_weight_train
            Optional sample weights for training batch
        sample_weight_test
            Optional sample weight for test sample

        Returns
        -------
        influence_maps
            Tensor of shape (batch_size, seq_len, emb_dim) or (batch_size, n_features)
        """
        batch_size = x_train_batch.shape[0]

        # Compute IHVP once for the test sample
        if len(x_test.shape) == 1:
            x_test = tf.expand_dims(x_test, 0)
        if len(y_test.shape) == 0:
            y_test = tf.expand_dims(y_test, 0)

        if sample_weight_test is not None:
            test_batch = (x_test, y_test, sample_weight_test)
        else:
            test_batch = (x_test, y_test)

        ihvp = self.ihvp_calculator._compute_ihvp_single_batch(test_batch)
        ihvp_vec = tf.reshape(ihvp, [-1])

        # Compute influence map for each training sample
        influence_maps = []

        for i in range(batch_size):
            x_train_i = x_train_batch[i:i+1]
            y_train_i = y_train_batch[i:i+1]
            sw_train_i = sample_weight_train[i:i+1] if sample_weight_train is not None else None

            influence_map = self._compute_single_tokenwise(
                x_train_i, y_train_i, ihvp_vec, sw_train_i
            )
            influence_maps.append(influence_map)

        return tf.concat(influence_maps, axis=0)

    def _compute_single_tokenwise(
            self,
            x_train: tf.Tensor,
            y_train: tf.Tensor,
            ihvp_vec: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Compute tokenwise influence for a single training sample given pre-computed IHVP.
        """
        x_train_var = tf.Variable(x_train, trainable=True)

        with tf.GradientTape() as outer_tape:
            outer_tape.watch(x_train_var)

            with tf.GradientTape() as inner_tape:
                predictions = self.model.model(x_train_var, training=True)

                if sample_weight is not None:
                    loss = self.model.loss_function(y_train, predictions, sample_weight)
                else:
                    loss = self.model.loss_function(y_train, predictions)

                if len(loss.shape) > 0:
                    loss = tf.reduce_sum(loss)

            trainable_vars = self.model.weights
            grad_train = inner_tape.gradient(loss, trainable_vars)
            grad_train_vec = tf.concat([tf.reshape(g, [-1]) for g in grad_train], axis=0)

            influence_scalar = tf.reduce_sum(ihvp_vec * grad_train_vec)

        influence_map = outer_tape.gradient(influence_scalar, x_train_var)

        return influence_map

    def compute_total_influence(
            self,
            x_train: tf.Tensor,
            y_train: tf.Tensor,
            x_test: tf.Tensor,
            y_test: tf.Tensor,
            sample_weight_train: Optional[tf.Tensor] = None,
            sample_weight_test: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Compute the total (scalar) influence of a training sample on a test sample.

        This is the sum of all token-wise influences, equivalent to the standard
        influence function: influence = -grad_test^T @ H^{-1} @ grad_train

        Parameters
        ----------
        x_train
            Training input tensor
        y_train
            Training label tensor
        x_test
            Test input tensor
        y_test
            Test label tensor
        sample_weight_train
            Optional sample weight for training sample
        sample_weight_test
            Optional sample weight for test sample

        Returns
        -------
        total_influence
            Scalar influence value
        """
        # Ensure batch dimension
        if len(x_train.shape) == 1:
            x_train = tf.expand_dims(x_train, 0)
        if len(x_test.shape) == 1:
            x_test = tf.expand_dims(x_test, 0)
        if len(y_train.shape) == 0:
            y_train = tf.expand_dims(y_train, 0)
        if len(y_test.shape) == 0:
            y_test = tf.expand_dims(y_test, 0)

        # Compute IHVP for test sample
        if sample_weight_test is not None:
            test_batch = (x_test, y_test, sample_weight_test)
        else:
            test_batch = (x_test, y_test)

        ihvp = self.ihvp_calculator._compute_ihvp_single_batch(test_batch)
        ihvp_vec = tf.reshape(ihvp, [-1])

        # Compute gradient for train sample
        with tf.GradientTape() as tape:
            predictions = self.model.model(x_train, training=True)

            if sample_weight_train is not None:
                loss = self.model.loss_function(y_train, predictions, sample_weight_train)
            else:
                loss = self.model.loss_function(y_train, predictions)

            if len(loss.shape) > 0:
                loss = tf.reduce_sum(loss)

        trainable_vars = self.model.weights
        grad_train = tape.gradient(loss, trainable_vars)
        grad_train_vec = tf.concat([tf.reshape(g, [-1]) for g in grad_train], axis=0)

        # Total influence = ihvp^T @ grad_train
        total_influence = tf.reduce_sum(ihvp_vec * grad_train_vec)

        return total_influence
