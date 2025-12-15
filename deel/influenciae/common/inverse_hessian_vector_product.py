# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing the different approaches for computing and approximating the
inverse-hessian-vector product: an essential block in the computation of influence
functions.
"""
from abc import ABC, abstractmethod
from enum import Enum
from argparse import ArgumentError

import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=E0611
from tensorflow.keras.models import Sequential  # pylint: disable=E0611

from .model_wrappers import BaseInfluenceModel, InfluenceModel

from ..types import Optional, Union, Tuple, List, Callable, Dict
from ..utils import assert_batched_dataset, conjugate_gradients_solve, map_to_device


class InverseHessianVectorProduct(ABC):
    """
    An interface for classes that perform hessian-vector products.

    Parameters
    ----------
    model
       A TF model following the InfluenceModel interface whose weights we wish to use for the calculation of
       these (inverse)-hessian-vector products.
    train_dataset
       A batched TF dataset containing the training dataset's point we wish to employ for the estimation of
       the hessian matrix.
    """
    def __init__(self, model: InfluenceModel, train_dataset: Optional[tf.data.Dataset]):
        if train_dataset is not None:
            self.cardinality = train_dataset.cardinality()

        self.model = model
        self.train_set = train_dataset


    @abstractmethod
    def _compute_ihvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-hessian-vector product of a group of points provided in the form of
        a batch of tensors.

        Parameters
        ----------
        group_batch
            A Tuple with a single batch of tensors containing the points of which we wish to
            compute the inverse-hessian-vector product
        use_gradient
            A boolean indicating whether the IHVP is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        ihvp
            A tensor containing a rank-1 tensor per input point.
        """
        raise NotImplementedError

    def compute_ihvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.data.Dataset:
        """
        Computes the inverse-hessian-vector product of a group of points.

        Parameters
        ----------
        group
            A TF dataset containing the group of points of which we wish to compute the
            inverse-hessian-vector product.
        use_gradient
            A boolean indicating whether the IHVP is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        ihvp
            A tensor containing one rank-1 tensor per input point
        """
        assert_batched_dataset(group)

        ihvp_dataset = group.map(lambda *single_batch: self._compute_ihvp_single_batch(single_batch, use_gradient))

        return ihvp_dataset

    @abstractmethod
    def _compute_hvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the hessian-vector product of a group of points.

        Parameters
        ----------
        group_batch
            A TF dataset containing the group of points of which we wish to compute the
            hessian-vector product.
        use_gradient
            A boolean indicating whether the hvp is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        hvp
            A tensor containing one rank-1 tensor per input point
        """
        raise NotImplementedError()

    def compute_hvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.data.Dataset:
        """
        Computes the hessian-vector product of a group of points.

        Parameters
        ----------
        group
            A TF dataset containing the group of points of which we wish to compute the
            hessian-vector product.
        use_gradient
            A boolean indicating whether the hvp is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        hvp
            A tensor containing one rank-1 tensor per input point
        """
        assert_batched_dataset(group)

        hvp_ds = group.map(lambda *single_batch: self._compute_hvp_single_batch(single_batch, use_gradient))

        return hvp_ds


class ExactIHVP(InverseHessianVectorProduct):
    """
    A class that performs the 'exact' computation of the inverse-hessian-vector product.
    As such, it will calculate the hessian of the provided model's loss wrt its parameters,
    compute its Moore-Penrose pseudo-inverse (for numerical stability) and the multiply it
    by the gradients.

    Notes
    -----
    To speed up the algorithm, the hessian matrix is calculated once at instantiation.

    For models with a considerable amount of weights, this implementation may be infeasible
    due to its O(n^2) complexity for the hessian, plus the O(n^3) for its inversion.
    If its memory consumption is too high, you should consider using the CGD approximation,
    or computing the hessian separately and initializing the ExactIHVP with this hessian while
    setting train_dataset to None. To expect it to work the hessian should be computed for
    the training_set.

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface.
    train_dataset
        The TF dataset, already batched and containing only the samples we wish to use for
        the computation of the hessian matrix. Either train_hessian or train_dataset should
        not be None but not both.
    train_hessian
        The estimated hessian matrix of the model's loss wrt its parameters computed with
        the samples used for the model's training. Either hessian or train_dataset should
        not be None but not both.
    """
    def __init__(
            self,
            model: InfluenceModel,
            train_dataset: Optional[tf.data.Dataset] = None,
            train_hessian: Optional[tf.Tensor] = None,
    ):
        super().__init__(model, train_dataset)
        if train_dataset is not None:
            nb_batch = tf.cast(train_dataset.cardinality(), dtype=tf.int32)
            self.inv_hessian = self._compute_inv_hessian(self.train_set, nb_batch)
            self.hessian = None
        elif train_hessian is not None:
            self.hessian = train_hessian
            self.inv_hessian = tf.linalg.pinv(train_hessian)
        else:
            raise ArgumentError("Either train_dataset or train_hessian can be set to None, but not both")

    def _compute_inv_hessian(self, dataset: tf.data.Dataset, nb_batch: int) -> tf.Tensor:
        """
        Compute the (pseudo)-inverse of the hessian matrix wrt to the model's parameters using
        backward-mode AD.

        Disclaimer
        ----------
        This implementation trades memory usage for speed, so it can be quite
        memory intensive, especially when dealing with big models.

        Parameters
        ----------
        dataset
            A TF dataset containing the whole or part of the training dataset for the
            computation of the inverse of the mean hessian matrix.

        Returns
        ----------
        inv_hessian
            A tf.Tensor with the resulting inverse hessian matrix
        """
        weights = self.model.weights

        hess = tf.zeros((self.model.nb_params, self.model.nb_params), dtype=dataset.element_spec[0].dtype)
        nb_elt = tf.constant(0, dtype=tf.int32)
        nb_batch_saw = tf.constant(0, dtype=tf.int32)
        iter_ds = iter(dataset)

        def hessian_sum(nb_elt, nb_batch_saw, hess):
            batch = next(iter_ds)
            nb_batch_saw += tf.constant(1, dtype=tf.int32)
            curr_nb_elt = tf.shape(batch[0])[0]
            nb_elt += curr_nb_elt
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_hess:
                tape_hess.watch(weights)
                grads = self.model.batch_jacobian_tensor(batch) # pylint: disable=W0212

            curr_hess = tape_hess.jacobian(
                    grads, weights
                    )

            curr_hess = [tf.reshape(h, shape=(len(grads), self.model.nb_params, -1)) for h in curr_hess]
            curr_hess = tf.concat(curr_hess, axis=-1)
            curr_hess = tf.reshape(curr_hess, shape=(len(grads), self.model.nb_params, -1))
            curr_hess = tf.reduce_sum(curr_hess, axis=0)
            hess += tf.cast(curr_hess, dtype=hess.dtype)

            return nb_elt, nb_batch_saw, hess

        nb_elt, _, hess = tf.while_loop(
            cond=lambda __, nb_batch_saw, _: nb_batch_saw < nb_batch,
            body=hessian_sum,
            loop_vars=[nb_elt, nb_batch_saw, hess]
        )

        hessian = hess / tf.cast(nb_elt, dtype=hess.dtype)

        return tf.linalg.pinv(hessian)

    def _compute_ihvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-hessian-vector product of a group of points provided in the form of
        a batch of tensors by computing the exact inverse hessian matrix and performing the product
        operation.

        Parameters
        ----------
        group_batch
            A Tuple with a single batch of tensors containing the points of which we wish to
            compute the inverse-hessian-vector product.
        use_gradient
            A boolean indicating whether the IHVP is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        ihvp
            A tensor containing a rank-1 tensor per input point.
        """
        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian_tensor(group_batch), (-1, self.model.nb_params))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        ihvp = tf.matmul(self.inv_hessian, tf.cast(grads, dtype=self.inv_hessian.dtype), transpose_b=True)
        return ihvp

    def _compute_hvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the hessian-vector product of a group of points provided in the form of a tuple
        of tensors by computing the hessian matrix and performing the product operation.

        Parameters
        ----------
        group_batch
            A Tuple with a single batch of tensors containing the points of which we wish to
            compute the hessian-vector product.
        use_gradient
            A boolean indicating whether the hvp is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        hvp
            A tensor containing one rank-1 tensor per input point
        """
        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian_tensor(group_batch), (-1, self.model.nb_params))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        hvp = tf.matmul(self.hessian, grads, transpose_b=True)

        return hvp

    def compute_hvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.data.Dataset:
        """
        Computes the hessian-vector product of a group of points provided in the form of a tuple
        of tensors by computing the hessian matrix and performing the product operation.

        Parameters
        ----------
        group
            A Tuple with a single batch of tensors containing the points of which we wish to
            compute the hessian-vector product.
        use_gradient
            A boolean indicating whether the hvp is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        hvp
            A tensor containing one rank-1 tensor per input point
        """
        if self.hessian is None:
            self.hessian = tf.linalg.pinv(self.inv_hessian)
        return super().compute_hvp(group, use_gradient)


class ForwardOverBackwardHVP:
    """
    A class for efficiently computing Hessian-vector products using forward-over-backward
    auto-differentiation.
    This module is used for the approximate IHVP calculations (CGD and LISSA).

    Parameters
    ----------
    model
        A TF model following the InfluenceModel interface.
    train_dataset
        A (batched) TF dataset with the data-points that will be used for the hessian.
    weights
        The target weights on which to calculate the HVP.
    """
    def __init__(
            self,
            model: BaseInfluenceModel,
            train_dataset: tf.data.Dataset,
            weights: Optional[List[tf.Tensor]] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.cardinality = train_dataset.cardinality()

        if weights is None:
            self.weights = model.weights
        else:
            self.weights = weights

    @staticmethod
    def _reshape_vector(grads: tf.Tensor, weights: tf.Tensor) -> List[tf.Tensor]:
        """
        Reshapes the gradient vector to the right shape for being input into the HVP computation.

        Parameters
        ----------
        grads
            A tensor with the computed gradients.
        weights
            A tensor with the target weights.

        Returns
        -------
        grads_reshape
            A list with the gradients in the right shape.
        """
        grads_reshape = []
        index = 0
        for w in weights:
            shape = tf.shape(w)
            size = tf.reduce_prod(shape)
            g = grads[index:(index + size)]
            grads_reshape.append(tf.reshape(g, shape))
            index += size
        return grads_reshape

    def _sub_call(
            self,
            x: tf.Tensor,
            feature_maps_hessian_current: tf.Tensor,
            y_hessian_current: tf.Tensor
    ) -> tf.Tensor:
        """
        Performs the hessian-vector product for a single feature map.

        Parameters
        ----------
        x
            The gradient vector to be multiplied by the hessian matrix.
        feature_maps_hessian_current
            The current feature map for the hessian calculation.
        y_hessian_current
            The label corresponding to the current feature map.

        Returns
        -------
        hessian_vector_product
            A tf.Tensor containing the result of the hessian-vector product for a given input point and one pair
            feature map-label.
        """
        with tf.autodiff.ForwardAccumulator(
                self.weights,
                # The "vector" in Hessian-vector product.
                x) as acc:
            # Use watch_accessed_variables=True for Keras 3.x compatibility
            with tf.GradientTape(persistent=False) as tape:
                loss = self.model.loss_function(y_hessian_current, self.model(feature_maps_hessian_current))
            backward = tape.jacobian(loss, self.weights)
        hessian_vector_product = acc.jvp(backward)

        hvp = [tf.reshape(hessian_vp, shape=(-1,)) for hessian_vp in hessian_vector_product]
        hvp = tf.concat(hvp, axis=0)

        weight = tf.cast(tf.shape(feature_maps_hessian_current)[0], dtype=hvp.dtype)

        hvp = hvp * weight

        return hvp

    def __call__(self, x_initial: tf.Tensor) -> tf.Tensor:
        """
        Computes the mean hessian-vector product for a given feature map over a set of points.

        Parameters
        ----------
        x_initial
            The point of the dataset over which this product will be computed

        Returns
        -------
        hessian_vector_product
            Tensor with the hessian-vector product
        """
        x = self._reshape_vector(x_initial, self.model.weights)

        hvp_init = tf.zeros((self.model.nb_params,), dtype=x_initial.dtype)
        dataset_iterator = iter(self.train_dataset)

        def body_func(i, hessian_vector_product, nb_hessian):
            features_block, labels_block = next(dataset_iterator)

            def batched_hvp(elt):
                f, l = elt
                hessian_product_current = self._sub_call(x, tf.expand_dims(f, axis=0), tf.expand_dims(l, axis=0))

                return hessian_product_current

            hessian_vector_product_inner = tf.reduce_sum(
                tf.map_fn(fn=batched_hvp, elems=[features_block, labels_block], fn_output_signature=x_initial.dtype),
                axis=0
            )

            hessian_vector_product += hessian_vector_product_inner
            return i + 1, hessian_vector_product, nb_hessian + tf.shape(features_block)[0]

        _, hessian_vector_product, nb_hessian = tf.while_loop(
            cond=lambda i, _, __: i < self.cardinality,
            body=body_func,
            loop_vars=[tf.constant(0, dtype=tf.int64), hvp_init, tf.constant(0, dtype=tf.int32)]
        )

        hessian_vector_product = tf.reshape(hessian_vector_product, (self.model.nb_params, 1)) / \
                                 tf.cast(nb_hessian, dtype=hessian_vector_product.dtype)

        return hessian_vector_product


class IterativeIHVP(InverseHessianVectorProduct):
    """
    A class that approximately computes inverse-hessian-vector products leveraging forward-over-backward
    automatic differentiation with an iterative procedure to estimate the product directly, without needing to
    calculate the hessian matrix or invert it.
    Notes
    -----
    It is ideal for models containing a considerable amount of parameters. It does however trade memory for
    speed, as the calculations for estimating the inverse hessian operator are repeated for each sample.
    Parameters
    ----------
    iterative_function
        The procedure to compute the inverse hessian product operation
    model
        The TF2.X model implementing the InfluenceModel interface
    extractor_layer
        An integer indicating the position of the last layer of the feature extraction network.
    train_dataset
        The TF dataset, already batched and containing only the samples we wish to use for the computation of the
        hessian matrix
    n_opt_iters
        The maximum amount of CGD iterations to perform when estimating the inverse-hessian
    feature_extractor
        If the feature extraction model is not Sequential, the full TF graph must be provided for the computation of
        the different feature maps.
    """
    def __init__(
            self,
            iterative_function,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: tf.data.Dataset,
            n_opt_iters: Optional[int] = 100,
            feature_extractor: Optional[Model] = None,
    ):
        super().__init__(model, train_dataset)
        self.n_opt_iters = n_opt_iters
        self._batch_shape_tensor = None
        self.extractor_layer = extractor_layer

        if feature_extractor is None:
            assert isinstance(model.model, Sequential)
            self.feature_extractor = tf.keras.Sequential(self.model.layers[:self.extractor_layer])
        else:
            assert isinstance(feature_extractor, Model)
            self.feature_extractor = feature_extractor

        self.train_set = self._compute_feature_map_dataset(self.train_set)  # extract the train set's features
        self.model = BaseInfluenceModel(
            tf.keras.Sequential(model.layers[extractor_layer:]),
            weights_to_watch=model.weights,
            loss_function=model.loss_function,
            weights_processed=True
        )  # model that predicts based on the extracted feature maps
        self.weights = self.model.weights
        self.hessian_vector_product = ForwardOverBackwardHVP(self.model, self.train_set, self.weights)
        self.iterative_function = iterative_function

    def batch_shape_tensor(self):
        """
        Return the batch shape of a tensor
        """
        return self._batch_shape_tensor

    def _compute_feature_map_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Extracts the feature maps for an entire dataset and creates a TF dataset associating them with
        their corresponding labels.
        Parameters
        ----------
        dataset
            The TF dataset whose feature maps we wish to extract using the model's first layers
        Returns
        -------
        feature_map_dataset
            A TF dataset with the pairs (feature_maps, labels), batched using the same batch_size as the one provided
            as input
        """
        feature_map_dataset = map_to_device(dataset, lambda x_batch, y: (self.feature_extractor(x_batch), y)).cache()

        if self._batch_shape_tensor is None:
            self._batch_shape_tensor = tf.shape(next(iter(feature_map_dataset))[0])

        return feature_map_dataset

    def _compute_ihvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-hessian-vector product of a group of points provided in the form of
        a batch of tensors by inverting the hessian-vector product that is calculated through
        forward-over-backward AD.
        Parameters
        ----------
        group_batch
            A Tuple with a single batch of tensors containing the points of which we wish to
            compute the inverse-hessian-vector product.
        use_gradient
            A boolean indicating whether the IHVP is with the gradients wrt to the loss of the
            points in group or with these vectors instead.
        Returns
        -------
        ihvp
            A tensor containing a rank-1 tensor per input point.
        """
        # Transform the dataset into a set of feature maps-labels
        if use_gradient:
            feature_maps = self.feature_extractor(group_batch[0])
            grads = self.model.batch_jacobian_tensor((feature_maps, *group_batch[1:]))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        # Compute the IHVP for each pair feature map-label
        def cgd_func(single_grad):
            inv_hessian_vect_product = self.iterative_function(self.hessian_vector_product,
                                                               tf.expand_dims(single_grad, axis=-1),
                                                               self.n_opt_iters)
            return inv_hessian_vect_product

        ihvp_list = tf.map_fn(fn=cgd_func, elems=grads)

        ihvp_list = tf.transpose(ihvp_list) if ihvp_list.shape[-1] != 1 \
            else tf.transpose(tf.squeeze(ihvp_list, axis=-1))

        return ihvp_list

    def _compute_hvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the hessian-vector product of a group of points provided in the form of a tuple
        of tensors through forward-over-backward AD.
        Parameters
        ----------
        group_batch
            A Tuple with a single batch of tensors containing the points of which we wish to
            compute the hessian-vector product.
        use_gradient
            A boolean indicating whether the hvp is with the gradients wrt to the loss of the
            points in group or with these vectors instead.
        Returns
        -------
        hvp
            A tensor containing one rank-1 tensor per input point
        """
        # Transform the dataset into a set of feature maps-labels
        if use_gradient:
            feature_maps = self.feature_extractor(group_batch[0])
            grads = self.model.batch_jacobian_tensor((feature_maps, *group_batch[1:]))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        # Compute the HVP for each pair features map - label
        def single_hvp(single_grad):
            hvp = self.hessian_vector_product(tf.expand_dims(single_grad, axis=-1))
            return hvp

        hvp_list = tf.map_fn(fn=single_hvp, elems=grads)

        hvp_list = tf.transpose(hvp_list) if hvp_list.shape[-1] != 1 else tf.transpose(tf.squeeze(hvp_list, axis=-1))

        return hvp_list


class ConjugateGradientDescentIHVP(IterativeIHVP):
    """
    A class that approximately computes inverse-hessian-vector products leveraging forward-over-backward
    automatic differentiation and Conjugate Gradient Descent to estimate the product directly, without needing to
    calculate the hessian matrix or invert it.
    Notes
    -----
    It is ideal for models containing a considerable amount of parameters. It does however trade memory for
    speed, as the calculations for estimating the inverse hessian operator are repeated for each sample.
    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface
    extractor_layer
        An integer indicating the position of the last layer of the feature extraction network.
    train_dataset
        The TF dataset, already batched and containing only the samples we wish to use for the computation of the
        hessian matrix
    n_opt_iters
        The maximum amount of CGD iterations to perform when estimating the inverse-hessian
    feature_extractor
        If the feature extraction model is not Sequential, the full TF graph must be provided for the computation of
        the different feature maps.
    """
    def __init__(
            self,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: tf.data.Dataset,
            n_opt_iters: Optional[int] = 100,
            feature_extractor: Optional[Model] = None,
    ):
        def iterative_function(operator, v, maxiter):  # pylint: disable=W0613
            return conjugate_gradients_solve(operator, v, x0=None, maxiter=self.n_opt_iters)
        super().__init__(iterative_function, model, extractor_layer, train_dataset, n_opt_iters, feature_extractor)


class LissaIHVP(IterativeIHVP):
    """
    A class that approximately computes inverse-hessian-vector products leveraging forward-over-backward
    automatic differentiation and lissa [https://arxiv.org/pdf/1703.04730.pdf , https://arxiv.org/pdf/1602.03943.pdf]
    to estimate the product directly, without needing to calculate the hessian matrix or invert it.

    [A^{-1}v]_{j+1} = v + (I - (A + d * I))[A^{-1}v]_j * v

    Notes
    -----
    It is ideal for models containing a considerable amount of parameters. It does however trade memory for
    speed, as the calculations for estimating the inverse hessian operator are repeated for each sample.

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface
    extractor_layer
        An integer indicating the position of the last layer of the feature extraction network.
    train_dataset
        The TF dataset, already batched and containing only the samples we wish to use for the computation of the
        hessian matrix
    n_opt_iters
        The maximum amount of CGD iterations to perform when estimating the inverse-hessian
    feature_extractor
        If the feature extraction model is not Sequential, the full TF graph must be provided for the computation of
        the different feature maps.
    damping
        A damping parameter to regularize a nearly singular operator.
    scale
        A rescaling factor to verify the hypothesis of norm(operator / scale) < 1.
    """
    def __init__(
            self,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: tf.data.Dataset,
            n_opt_iters: Optional[int] = 100,
            feature_extractor: Optional[Model] = None,
            damping: float = 1e-4,
            scale: float = 10.
    ):
        super().__init__(self.lissa, model, extractor_layer, train_dataset, n_opt_iters, feature_extractor)
        self.damping = tf.convert_to_tensor(damping, dtype=tf.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=tf.float32)

    def lissa(self, operator: Callable, v: tf.Tensor, maxiter: int):
        """
        Performs the Linear time Stochastic Second-order Algorithm (LiSSA) optimization procedure to solve
        a problem of the shape Ax = b by iterating as follows:

            [A^{-1}v]_{j+1} = v + (I - (A + d * I))[A^{-1}v]_j * v

        Parameters
        ----------
        operator
            The operator that transforms the input vector v into Av
        v
            The vector v of the problem
        maxiter
            Number of iterations of the algorithm

        Returns
        -------
        ihvp_result
            A tensor containing inv(A)v
        """
        _, ihvp_result = tf.while_loop(lambda index, ihvp: index < maxiter,
                                       lambda index, ihvp: (index + 1,
                                                            v + tf.cast(1. - self.damping, dtype=tf.float32) * ihvp -
                                                            operator(ihvp) / self.scale),
                                       [tf.constant(0, dtype=tf.int32), v])
        ihvp_result /= self.scale

        return ihvp_result


class FisherIHVP(InverseHessianVectorProduct):
    """
    A class that computes the inverse-hessian-vector product using the Fisher Information Matrix
    (Gauss-Newton approximation) instead of the exact Hessian.

    The Fisher/Gauss-Newton approximation is:
        F = E[J^T J]  where J = ∂f/∂θ (Jacobian of model outputs w.r.t. parameters)

    This approximation:
    - Is always positive semi-definite (unlike the exact Hessian)
    - Is faster to compute (no double backprop needed)
    - Is equivalent to the Hessian when the model is at a local minimum with zero residuals

    The inversion is done via LDL decomposition with Tikhonov regularization:
        (F + λI)^{-1} v

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface.
    train_dataset
        The TF dataset, already batched, containing the samples for computing the Fisher matrix.
    damping
        Tikhonov regularization parameter λ for numerical stability (default: 1e-4).
    train_fisher
        Optional pre-computed Fisher matrix. If provided, train_dataset can be None.
    extractor_layer
        Optional integer indicating the position of the last layer of the feature extraction network.
        If provided, the model will be split into feature extractor and head.
    feature_extractor
        Optional pre-built feature extractor model. If not provided but extractor_layer is,
        will be built from model.layers[:extractor_layer].

    Notes
    -----
    Memory complexity: O(p²) where p = number of parameters
    Time complexity: O(n*p*c) for computing F, O(p³) for inversion
    where n = number of samples, c = number of outputs

    Unlike ExactIHVP which computes ∂²L/∂θ² (requires double backprop),
    this computes J^T J (only single backprop + matrix product).
    """
    def __init__(
            self,
            model: InfluenceModel,
            train_dataset: Optional[tf.data.Dataset] = None,
            damping: float = 1e-4,
            train_fisher: Optional[tf.Tensor] = None,
            extractor_layer: Optional[Union[int, str]] = None,
            feature_extractor: Optional[Model] = None,
    ):
        super().__init__(model, train_dataset)
        self.damping = damping
        self.extractor_layer = extractor_layer
        self._batch_shape_tensor = None

        # Handle feature extractor setup if extractor_layer is provided
        if extractor_layer is not None:
            if feature_extractor is None:
                self.feature_extractor = tf.keras.Sequential(model.model.layers[:extractor_layer])
            else:
                self.feature_extractor = feature_extractor

            # Transform training dataset to feature maps
            if train_dataset is not None:
                train_dataset = self._compute_feature_map_dataset(train_dataset)

            # Create a new model that operates on feature maps
            self.model = BaseInfluenceModel(
                tf.keras.Sequential(model.model.layers[extractor_layer:]),
                weights_to_watch=model.weights,
                loss_function=model.loss_function,
                weights_processed=True
            )
        else:
            self.feature_extractor = None

        if train_dataset is not None:
            self.fisher = self._compute_fisher(train_dataset)
            self.inv_fisher = self._compute_inv_fisher(self.fisher, damping)
        elif train_fisher is not None:
            self.fisher = train_fisher
            self.inv_fisher = self._compute_inv_fisher(train_fisher, damping)
        else:
            raise ArgumentError("Either train_dataset or train_fisher must be provided")

    def _compute_feature_map_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Extracts the feature maps for an entire dataset.

        Parameters
        ----------
        dataset
            The TF dataset whose feature maps we wish to extract.

        Returns
        -------
        feature_map_dataset
            A TF dataset with the pairs (feature_maps, labels).
        """
        feature_map_dataset = map_to_device(dataset, lambda x_batch, y: (self.feature_extractor(x_batch), y)).cache()

        if self._batch_shape_tensor is None:
            self._batch_shape_tensor = tf.shape(next(iter(feature_map_dataset))[0])

        return feature_map_dataset

    def _compute_fisher(self, dataset: tf.data.Dataset) -> tf.Tensor:
        """
        Compute the Fisher Information Matrix (Gauss-Newton approximation).

        F = (1/n) Σᵢ Jᵢ^T Jᵢ

        where Jᵢ = ∂L_i/∂θ is the gradient of the loss for sample i.

        Unlike the exact Hessian which requires ∂²L/∂θ² (jacobian of jacobian),
        Fisher only needs the outer product of gradients: J^T J.

        Parameters
        ----------
        dataset
            Batched TF dataset for computing the Fisher matrix.

        Returns
        -------
        fisher
            The Fisher Information Matrix of shape (nb_params, nb_params).
        """
        nb_params = self.model.nb_params

        # Initialize Fisher matrix
        fisher = tf.zeros((nb_params, nb_params), dtype=tf.float32)
        n_samples = 0

        for batch in dataset:
            # Get per-sample gradients (Jacobian of loss w.r.t. parameters)
            # Shape: (batch_size, nb_params)
            grads = self.model.batch_jacobian_tensor(batch)
            grads = tf.reshape(grads, (-1, nb_params))
            grads = tf.cast(grads, dtype=tf.float32)

            # Replace NaN/Inf with zeros for numerical stability
            grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))

            batch_size = tf.shape(grads)[0]
            n_samples += batch_size

            # Fisher = sum of outer products: J^T @ J
            # This is the key difference from ExactIHVP which does jacobian(jacobian(...))
            fisher += tf.matmul(grads, grads, transpose_a=True)

        # Average over samples
        fisher = fisher / tf.cast(n_samples, dtype=fisher.dtype)

        # Ensure Fisher is symmetric and finite
        fisher = (fisher + tf.transpose(fisher)) / 2.0
        fisher = tf.where(tf.math.is_finite(fisher), fisher, tf.zeros_like(fisher))

        return fisher

    @staticmethod
    def _compute_inv_fisher(fisher: tf.Tensor, damping: float) -> tf.Tensor:
        """
        Compute the inverse of the Fisher matrix with Tikhonov regularization.

        Uses LDL decomposition for numerical stability (similar to nngeometry).

        Parameters
        ----------
        fisher
            The Fisher Information Matrix.
        damping
            Regularization parameter λ.

        Returns
        -------
        inv_fisher
            The regularized inverse: (F + λI)^{-1}
        """
        n = tf.shape(fisher)[0]
        regularized_fisher = fisher + damping * tf.eye(n, dtype=fisher.dtype)

        # Use pseudo-inverse for better numerical stability
        # This handles rank-deficient or near-singular matrices gracefully
        inv_fisher = tf.linalg.pinv(regularized_fisher)

        # Ensure result is finite
        inv_fisher = tf.where(tf.math.is_finite(inv_fisher), inv_fisher, tf.zeros_like(inv_fisher))

        return inv_fisher

    def solve(self, v: tf.Tensor, damping: Optional[float] = None) -> tf.Tensor:
        """
        Solve (F + λI)x = v for x without explicitly computing the inverse.

        This is more numerically stable than computing F^{-1} @ v.

        Parameters
        ----------
        v
            The vector to solve for, shape (nb_params,) or (nb_params, k).
        damping
            Optional damping override (uses self.damping if None).

        Returns
        -------
        x
            The solution x = (F + λI)^{-1} v
        """
        if damping is None:
            damping = self.damping

        n = tf.shape(self.fisher)[0]
        regularized_fisher = self.fisher + damping * tf.eye(n, dtype=self.fisher.dtype)

        # Solve via Cholesky
        v = tf.cast(v, dtype=regularized_fisher.dtype)
        if len(v.shape) == 1:
            v = tf.expand_dims(v, axis=-1)

        try:
            chol = tf.linalg.cholesky(regularized_fisher)
            solution = tf.linalg.cholesky_solve(chol, v)
        except tf.errors.InvalidArgumentError:
            solution = tf.linalg.lstsq(regularized_fisher, v)

        return tf.squeeze(solution)

    def _compute_ihvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-Fisher-vector product for a batch of points.

        Parameters
        ----------
        group_batch
            A tuple containing the batch of inputs and labels.
        use_gradient
            If True, compute gradients of the loss; if False, use the input directly as vectors.

        Returns
        -------
        ihvp
            The inverse-Fisher-vector product, shape (nb_params, batch_size).
        """
        # Handle feature extraction if configured
        if self.feature_extractor is not None and use_gradient:
            feature_maps = self.feature_extractor(group_batch[0])
            batch_for_grad = (feature_maps, *group_batch[1:])
        else:
            batch_for_grad = group_batch

        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian_tensor(batch_for_grad), (-1, self.model.nb_params))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        # Replace NaN/Inf with zeros for numerical stability
        grads = tf.cast(grads, dtype=self.inv_fisher.dtype)
        grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))

        # F^{-1} @ grads^T
        ihvp = tf.matmul(self.inv_fisher, grads, transpose_b=True)

        # Ensure result is finite
        ihvp = tf.where(tf.math.is_finite(ihvp), ihvp, tf.zeros_like(ihvp))

        return ihvp

    def _compute_hvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the Fisher-vector product for a batch of points.

        Parameters
        ----------
        group_batch
            A tuple containing the batch of inputs and labels.
        use_gradient
            If True, compute gradients of the loss; if False, use the input directly as vectors.

        Returns
        -------
        hvp
            The Fisher-vector product, shape (nb_params, batch_size).
        """
        # Handle feature extraction if configured
        if self.feature_extractor is not None and use_gradient:
            feature_maps = self.feature_extractor(group_batch[0])
            batch_for_grad = (feature_maps, *group_batch[1:])
        else:
            batch_for_grad = group_batch

        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian_tensor(batch_for_grad), (-1, self.model.nb_params))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        grads = tf.cast(grads, dtype=self.fisher.dtype)
        grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))

        hvp = tf.matmul(self.fisher, grads, transpose_b=True)
        hvp = tf.where(tf.math.is_finite(hvp), hvp, tf.zeros_like(hvp))

        return hvp

class KFACIHVP(InverseHessianVectorProduct):
    """
    Inverse Hessian-Vector Product calculator using Kronecker-Factored Approximate Curvature (KFAC).

    KFAC approximates the Fisher Information Matrix as a block-diagonal matrix where each block
    (corresponding to one layer) is approximated as a Kronecker product of two smaller matrices:

    H_layer ≈ G ⊗ A

    where:
    - A is the input covariance matrix: A = E[a @ a^T] (size: in_features x in_features)
    - G is the output gradient covariance: G = E[g @ g^T] (size: out_features x out_features)

    This dramatically reduces memory from O(p²) to O(Σ(in_i² + out_i²)) where p is total params.

    The key insight is that for a linear layer y = Wx + b:
    - The gradient w.r.t. W is: ∂L/∂W = g @ a^T (outer product)
    - The Fisher block is: E[(g ⊗ a)(g ⊗ a)^T] = E[gg^T] ⊗ E[aa^T] = G ⊗ A

    Parameters
    ----------
    model
        An InfluenceModel wrapping the TensorFlow model.
    train_dataset
        A batched TF dataset for computing the KFAC factors.
    damping
        Regularization parameter λ for numerical stability.

    Attributes
    ----------
    kfac_blocks
        Dictionary mapping layer index to (A, G) Kronecker factor pairs.
    layer_shapes
        Dictionary mapping layer index to (in_features, out_features) tuples.

    Notes
    -----
    Memory complexity: O(Σ(in_i² + out_i²)) vs O(p²) for full Fisher
    Time complexity: O(n * Σ(in_i * out_i)) for computing factors
    Solve complexity: O(Σ(in_i³ + out_i³)) for Cholesky solves

    References
    ----------
    Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored
    approximate curvature. ICML.
    """

    def __init__(
            self,
            model: InfluenceModel,
            train_dataset: Optional[tf.data.Dataset] = None,
            damping: float = 1e-3,
    ):
        super().__init__(model, train_dataset)
        self.damping = damping

        # Extract layer information from the model
        self.layer_info = self._extract_layer_info()

        if train_dataset is not None:
            self.kfac_blocks = self._compute_kfac_blocks(train_dataset)
        else:
            raise ArgumentError("train_dataset must be provided for KFAC")

    def _extract_layer_info(self) -> List[dict]:
        """
        Extract information about Dense layers in the model.

        Returns
        -------
        layer_info
            List of dictionaries with layer information (weights, shapes, etc.)
        """
        layer_info = []
        weight_idx = 0

        for layer in self.model.model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                if len(weights) >= 1:
                    kernel = weights[0]  # Shape: (in_features, out_features)
                    has_bias = len(weights) > 1

                    in_features = kernel.shape[0]
                    out_features = kernel.shape[1]

                    # Count parameters for this layer
                    n_params = in_features * out_features
                    if has_bias:
                        n_params += out_features

                    layer_info.append({
                        'layer': layer,
                        'in_features': in_features,
                        'out_features': out_features,
                        'has_bias': has_bias,
                        'weight_start_idx': weight_idx,
                        'n_params': n_params,
                    })
                    weight_idx += n_params

        return layer_info

    def _compute_kfac_blocks(self, dataset: tf.data.Dataset) -> Dict[int, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Compute KFAC factors (A, G) for each layer.

        For each Dense layer:
        - A = (1/n) Σ a_i @ a_i^T  (input covariance, with bias augmentation)
        - G = (1/n) Σ g_i @ g_i^T  (output gradient covariance)

        Parameters
        ----------
        dataset
            Batched TF dataset for computing the factors.

        Returns
        -------
        kfac_blocks
            Dictionary mapping layer index to (A, G) tuples.
        """
        # Initialize accumulators for each layer
        A_accum = {}
        G_accum = {}
        n_samples = 0

        for layer_idx, info in enumerate(self.layer_info):
            in_dim = info['in_features'] + (1 if info['has_bias'] else 0)
            out_dim = info['out_features']
            A_accum[layer_idx] = tf.zeros((in_dim, in_dim), dtype=tf.float32)
            G_accum[layer_idx] = tf.zeros((out_dim, out_dim), dtype=tf.float32)

        # Build intermediate models to get activations
        layer_inputs = []
        layer_outputs = []

        for info in self.layer_info:
            layer = info['layer']
            # Get the input to this layer
            layer_input_model = tf.keras.Model(
                inputs=self.model.model.input,
                outputs=layer.input
            )
            layer_inputs.append(layer_input_model)

        # Process dataset
        for batch in dataset:
            inputs, labels, sample_weight = self.model.process_batch_for_loss_fn(batch)
            batch_size = tf.shape(inputs)[0]
            n_samples += batch_size

            # Forward pass with gradient tape to get per-layer gradients
            with tf.GradientTape(persistent=True) as tape:
                # Get activations at each layer input
                activations = [model(inputs) for model in layer_inputs]

                # Watch activations to compute gradients w.r.t. them
                for act in activations:
                    tape.watch(act)

                # Full forward pass
                output = self.model.model(inputs)
                loss_per_sample = self.model.loss_function(labels, output, sample_weight)
                loss = tf.reduce_sum(loss_per_sample)

            # Compute gradients w.r.t. each layer's output (pre-activation gradient)
            for layer_idx, info in enumerate(self.layer_info):
                layer = info['layer']
                activation = activations[layer_idx]

                # Compute output of this layer
                layer_output = layer(activation)

                # Gradient of loss w.r.t. layer output (this is 'g' in KFAC)
                grad_output = tape.gradient(loss, layer_output)

                if grad_output is None:
                    continue

                # A factor: input covariance
                # Augment with 1 for bias
                a = activation
                if info['has_bias']:
                    ones = tf.ones((batch_size, 1), dtype=a.dtype)
                    a = tf.concat([a, ones], axis=1)

                # Accumulate A = a^T @ a
                A_accum[layer_idx] += tf.matmul(a, a, transpose_a=True)

                # G factor: output gradient covariance
                g = grad_output
                # Accumulate G = g^T @ g
                G_accum[layer_idx] += tf.matmul(g, g, transpose_a=True)

            del tape

        # Average and create blocks
        kfac_blocks = {}
        n_samples_float = tf.cast(n_samples, tf.float32)

        for layer_idx in range(len(self.layer_info)):
            A = A_accum[layer_idx] / n_samples_float
            G = G_accum[layer_idx] / n_samples_float

            # Ensure symmetric and finite
            A = (A + tf.transpose(A)) / 2.0
            G = (G + tf.transpose(G)) / 2.0
            A = tf.where(tf.math.is_finite(A), A, tf.zeros_like(A))
            G = tf.where(tf.math.is_finite(G), G, tf.zeros_like(G))

            kfac_blocks[layer_idx] = (A, G)

        return kfac_blocks

    def _compute_pi_correction(self, A: tf.Tensor, G: tf.Tensor) -> float:
        """
        Compute pi correction factor for balanced damping.

        pi = sqrt((trace(A) * dim(G)) / (trace(G) * dim(A)))

        This balances the regularization between the two factors.

        Parameters
        ----------
        A
            Input covariance factor.
        G
            Output gradient covariance factor.

        Returns
        -------
        pi
            The correction factor.
        """
        trace_A = tf.linalg.trace(A)
        trace_G = tf.linalg.trace(G)
        dim_A = tf.cast(tf.shape(A)[0], tf.float32)
        dim_G = tf.cast(tf.shape(G)[0], tf.float32)

        # Avoid division by zero
        trace_G = tf.maximum(trace_G, 1e-10)
        dim_A = tf.maximum(dim_A, 1.0)

        pi = tf.sqrt((trace_A * dim_G) / (trace_G * dim_A))
        pi = tf.clip_by_value(pi, 1e-5, 1e5)  # Prevent extreme values

        return float(pi.numpy())

    def _solve_kfac_block(
            self,
            A: tf.Tensor,
            G: tf.Tensor,
            v_block: tf.Tensor,
            damping: float
    ) -> tf.Tensor:
        """
        Solve (G ⊗ A + λI) @ x = v for x using the Kronecker structure.

        Using the property: (G ⊗ A)^{-1} = G^{-1} ⊗ A^{-1}
        We solve: x = (A + sqrt(λ/pi) I)^{-1} @ V @ (G + sqrt(λ*pi) I)^{-T}

        where V is v_block reshaped to (in_features, out_features).

        Parameters
        ----------
        A
            Input covariance factor.
        G
            Output gradient covariance factor.
        v_block
            Vector block for this layer (flattened).
        damping
            Regularization parameter.

        Returns
        -------
        x_block
            Solution vector (flattened).
        """
        in_dim = A.shape[0]
        out_dim = G.shape[0]

        # Compute pi correction
        pi = self._compute_pi_correction(A, G)

        # Add damping with pi correction
        sqrt_damping = tf.sqrt(damping)
        A_reg = A + (sqrt_damping / pi) * tf.eye(in_dim, dtype=A.dtype)
        G_reg = G + (sqrt_damping * pi) * tf.eye(out_dim, dtype=G.dtype)

        # Reshape v to matrix form (in_features, out_features)
        V = tf.reshape(v_block, (in_dim, out_dim))

        # Solve using two linear solves
        # First solve: A_reg @ X_temp = V
        X_temp = tf.linalg.solve(A_reg, V)

        # Second solve: G_reg @ X^T = X_temp^T => X = (G_reg^{-1} @ X_temp^T)^T
        X = tf.transpose(tf.linalg.solve(G_reg, tf.transpose(X_temp)))

        # Flatten back
        return tf.reshape(X, [-1])

    def _compute_ihvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-KFAC-vector product for a batch of points.

        Parameters
        ----------
        group_batch
            A tuple containing the batch of inputs and labels.
        use_gradient
            If True, compute gradients of the loss; if False, use the input directly as vectors.

        Returns
        -------
        ihvp
            The inverse-KFAC-vector product, shape (nb_params, batch_size).
        """
        if use_gradient:
            grads = self.model.batch_jacobian_tensor(group_batch)
            grads = tf.reshape(grads, (-1, self.model.nb_params))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        grads = tf.cast(grads, dtype=tf.float32)
        grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))

        batch_size = tf.shape(grads)[0]
        ihvp_list = []

        # Process each sample in the batch
        for i in range(batch_size):
            grad = grads[i]
            ihvp_sample = []

            # Solve for each layer block
            param_idx = 0
            for layer_idx, info in enumerate(self.layer_info):
                n_params = info['n_params']
                v_block = grad[param_idx:param_idx + n_params]

                A, G = self.kfac_blocks[layer_idx]
                x_block = self._solve_kfac_block(A, G, v_block, self.damping)

                ihvp_sample.append(x_block)
                param_idx += n_params

            ihvp_list.append(tf.concat(ihvp_sample, axis=0))

        ihvp = tf.stack(ihvp_list, axis=0)
        ihvp = tf.where(tf.math.is_finite(ihvp), ihvp, tf.zeros_like(ihvp))

        return tf.transpose(ihvp)

    def _compute_hvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the KFAC-vector product for a batch of points.

        This is the forward operation: (G ⊗ A) @ v for each layer block.

        Parameters
        ----------
        group_batch
            A tuple containing the batch of inputs and labels.
        use_gradient
            If True, compute gradients of the loss; if False, use the input directly as vectors.

        Returns
        -------
        hvp
            The KFAC-vector product, shape (nb_params, batch_size).
        """
        if use_gradient:
            grads = self.model.batch_jacobian_tensor(group_batch)
            grads = tf.reshape(grads, (-1, self.model.nb_params))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        grads = tf.cast(grads, dtype=tf.float32)
        grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))

        batch_size = tf.shape(grads)[0]
        hvp_list = []

        # Process each sample in the batch
        for i in range(batch_size):
            grad = grads[i]
            hvp_sample = []

            # Apply KFAC product for each layer block
            param_idx = 0
            for layer_idx, info in enumerate(self.layer_info):
                n_params = info['n_params']
                v_block = grad[param_idx:param_idx + n_params]

                A, G = self.kfac_blocks[layer_idx]
                x_block = self._apply_kfac_block(A, G, v_block)

                hvp_sample.append(x_block)
                param_idx += n_params

            hvp_list.append(tf.concat(hvp_sample, axis=0))

        hvp = tf.stack(hvp_list, axis=0)
        hvp = tf.where(tf.math.is_finite(hvp), hvp, tf.zeros_like(hvp))

        return tf.transpose(hvp)

    def _apply_kfac_block(
            self,
            A: tf.Tensor,
            G: tf.Tensor,
            v_block: tf.Tensor,
    ) -> tf.Tensor:
        """
        Apply (G ⊗ A) @ v for a single layer block.

        Using the property: (G ⊗ A) @ vec(V) = vec(A @ V @ G^T)

        Parameters
        ----------
        A
            Input covariance factor.
        G
            Output gradient covariance factor.
        v_block
            Vector block for this layer (flattened).

        Returns
        -------
        result_block
            Result vector (flattened).
        """
        in_dim = A.shape[0]
        out_dim = G.shape[0]

        # Reshape v to matrix form (in_features, out_features)
        V = tf.reshape(v_block, (in_dim, out_dim))

        # Apply: A @ V @ G^T
        result = tf.matmul(A, tf.matmul(V, G, transpose_b=True))

        # Flatten back
        return tf.reshape(result, [-1])


class EKFACIHVP(KFACIHVP):
    """
    Inverse Hessian-Vector Product calculator using Eigenvalue-corrected KFAC (EKFAC).

    EKFAC improves upon KFAC by computing the true eigenvalues of the Fisher matrix
    in the Kronecker-factored eigenbasis, rather than assuming they are products
    of the individual factor eigenvalues.

    The key insight is:
    1. Eigendecompose A = U_A Λ_A U_A^T and G = U_G Λ_G U_G^T
    2. The eigenvectors of G ⊗ A are columns of U_G ⊗ U_A (Kronecker product)
    3. But eigenvalues are NOT λ_G × λ_A in general
    4. EKFAC computes the true eigenvalues by projecting gradients to the eigenbasis

    The corrected eigenvalues are computed as:
    Λ_corrected[i,j] = (1/n) Σ_samples (g_kfe[i])² × (a_kfe[j])²

    where g_kfe = U_G^T @ g and a_kfe = U_A^T @ a are the projections to eigenbasis.

    Parameters
    ----------
    model
        An InfluenceModel wrapping the TensorFlow model.
    train_dataset
        A batched TF dataset for computing the KFAC factors and eigenvalue corrections.
    damping
        Regularization parameter λ for numerical stability.
    update_eigen
        Whether to compute the eigenvalue corrections (True for full EKFAC).

    Attributes
    ----------
    evecs
        Dictionary mapping layer index to (U_A, U_G) eigenvector matrices.
    evals
        Dictionary mapping layer index to 2D eigenvalue matrices (out_dim, in_dim).

    Notes
    -----
    EKFAC provides better approximation than KFAC at the cost of computing
    eigendecompositions and an additional pass through the data.

    References
    ----------
    George, T., et al. (2018). Fast Approximate Natural Gradient Descent
    in a Kronecker-factored Eigenbasis. NeurIPS.
    """

    def __init__(
            self,
            model: InfluenceModel,
            train_dataset: Optional[tf.data.Dataset] = None,
            damping: float = 1e-3,
            update_eigen: bool = True,
    ):
        # Initialize KFAC first
        super().__init__(model, train_dataset, damping)

        # Compute eigendecompositions of KFAC factors
        self.evecs = {}
        self.evals = {}

        for layer_idx, (A, G) in self.kfac_blocks.items():
            # Eigendecompose A and G
            evals_A, evecs_A = tf.linalg.eigh(A)
            evals_G, evecs_G = tf.linalg.eigh(G)

            # Ensure non-negative eigenvalues (numerical issues)
            evals_A = tf.maximum(evals_A, 0.0)
            evals_G = tf.maximum(evals_G, 0.0)

            self.evecs[layer_idx] = (evecs_A, evecs_G)

            # Initial eigenvalues: outer product of factor eigenvalues
            # Shape: (out_dim, in_dim)
            self.evals[layer_idx] = tf.expand_dims(evals_G, 1) * tf.expand_dims(evals_A, 0)

        # Update eigenvalues with true values if requested
        if update_eigen and train_dataset is not None:
            self._update_eigenvalues(train_dataset)

    def _update_eigenvalues(self, dataset: tf.data.Dataset) -> None:
        """
        Compute corrected eigenvalues by projecting gradients to the eigenbasis.

        For each sample, compute:
        a_kfe = U_A^T @ a  (project input to eigenbasis)
        g_kfe = U_G^T @ g  (project gradient to eigenbasis)

        Then: Λ[i,j] = (1/n) Σ (g_kfe[i])² × (a_kfe[j])²

        Parameters
        ----------
        dataset
            Batched TF dataset for computing eigenvalue corrections.
        """
        # Initialize accumulators
        evals_accum = {}
        for layer_idx, info in enumerate(self.layer_info):
            out_dim = info['out_features']
            in_dim = info['in_features'] + (1 if info['has_bias'] else 0)
            evals_accum[layer_idx] = tf.zeros((out_dim, in_dim), dtype=tf.float32)

        n_samples = 0

        # Build intermediate models
        layer_inputs = []
        for info in self.layer_info:
            layer = info['layer']
            layer_input_model = tf.keras.Model(
                inputs=self.model.model.input,
                outputs=layer.input
            )
            layer_inputs.append(layer_input_model)

        # Process dataset
        for batch in dataset:
            inputs, labels, sample_weight = self.model.process_batch_for_loss_fn(batch)
            batch_size = tf.shape(inputs)[0]
            n_samples += batch_size

            with tf.GradientTape(persistent=True) as tape:
                activations = [model(inputs) for model in layer_inputs]
                for act in activations:
                    tape.watch(act)

                output = self.model.model(inputs)
                loss_per_sample = self.model.loss_function(labels, output, sample_weight)
                loss = tf.reduce_sum(loss_per_sample)

            for layer_idx, info in enumerate(self.layer_info):
                layer = info['layer']
                activation = activations[layer_idx]
                layer_output = layer(activation)

                grad_output = tape.gradient(loss, layer_output)
                if grad_output is None:
                    continue

                evecs_A, evecs_G = self.evecs[layer_idx]

                # Augment activation with bias term
                a = activation
                if info['has_bias']:
                    ones = tf.ones((batch_size, 1), dtype=a.dtype)
                    a = tf.concat([a, ones], axis=1)

                g = grad_output

                # Project to eigenbasis
                a_kfe = tf.matmul(a, evecs_A)  # (batch, in_dim)
                g_kfe = tf.matmul(g, evecs_G)  # (batch, out_dim)

                # Compute eigenvalue contributions: (g_kfe²) ⊗ (a_kfe²)
                # For each sample: outer product of squared projections
                # Sum over batch
                a_kfe_sq = tf.square(a_kfe)  # (batch, in_dim)
                g_kfe_sq = tf.square(g_kfe)  # (batch, out_dim)

                # Accumulate: Σ_batch g²[:, i] * a²[:, j] for all i,j
                evals_contrib = tf.matmul(g_kfe_sq, a_kfe_sq, transpose_a=True)  # (out_dim, in_dim)
                evals_accum[layer_idx] += evals_contrib

            del tape

        # Average eigenvalues
        n_samples_float = tf.cast(n_samples, tf.float32)
        for layer_idx in range(len(self.layer_info)):
            self.evals[layer_idx] = evals_accum[layer_idx] / n_samples_float
            # Ensure non-negative
            self.evals[layer_idx] = tf.maximum(self.evals[layer_idx], 0.0)

    def _solve_ekfac_block(
            self,
            evecs_A: tf.Tensor,
            evecs_G: tf.Tensor,
            evals: tf.Tensor,
            v_block: tf.Tensor,
            damping: float
    ) -> tf.Tensor:
        """
        Solve the EKFAC system in the eigenbasis.

        In the eigenbasis, the system is diagonal:
        (Λ + λI) @ x_kfe = v_kfe

        So: x_kfe = v_kfe / (Λ + λ)

        Then transform back: x = (U_A ⊗ U_G) @ x_kfe

        Parameters
        ----------
        evecs_A
            Eigenvectors of A factor.
        evecs_G
            Eigenvectors of G factor.
        evals
            2D eigenvalue matrix (out_dim, in_dim).
        v_block
            Vector block for this layer (flattened).
        damping
            Regularization parameter.

        Returns
        -------
        x_block
            Solution vector (flattened).
        """
        in_dim = evecs_A.shape[0]
        out_dim = evecs_G.shape[0]

        # Reshape v to matrix form
        V = tf.reshape(v_block, (in_dim, out_dim))

        # Project to eigenbasis: V_kfe = U_A^T @ V @ U_G
        V_kfe = tf.matmul(tf.matmul(evecs_A, V, transpose_a=True), evecs_G)

        # Solve in eigenbasis (element-wise division)
        # Note: evals is (out_dim, in_dim), V_kfe is (in_dim, out_dim)
        evals_T = tf.transpose(evals)  # (in_dim, out_dim)
        X_kfe = V_kfe / (evals_T + damping)

        # Handle numerical issues
        X_kfe = tf.where(tf.math.is_finite(X_kfe), X_kfe, tf.zeros_like(X_kfe))

        # Project back: X = U_A @ X_kfe @ U_G^T
        X = tf.matmul(tf.matmul(evecs_A, X_kfe), evecs_G, transpose_b=True)

        return tf.reshape(X, [-1])

    def _compute_ihvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-EKFAC-vector product for a batch of points.

        Parameters
        ----------
        group_batch
            A tuple containing the batch of inputs and labels.
        use_gradient
            If True, compute gradients of the loss; if False, use the input directly as vectors.

        Returns
        -------
        ihvp
            The inverse-EKFAC-vector product, shape (nb_params, batch_size).
        """
        if use_gradient:
            grads = self.model.batch_jacobian_tensor(group_batch)
            grads = tf.reshape(grads, (-1, self.model.nb_params))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        grads = tf.cast(grads, dtype=tf.float32)
        grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))

        batch_size = tf.shape(grads)[0]
        ihvp_list = []

        for i in range(batch_size):
            grad = grads[i]
            ihvp_sample = []

            param_idx = 0
            for layer_idx, info in enumerate(self.layer_info):
                n_params = info['n_params']
                v_block = grad[param_idx:param_idx + n_params]

                evecs_A, evecs_G = self.evecs[layer_idx]
                evals = self.evals[layer_idx]

                x_block = self._solve_ekfac_block(evecs_A, evecs_G, evals, v_block, self.damping)

                ihvp_sample.append(x_block)
                param_idx += n_params

            ihvp_list.append(tf.concat(ihvp_sample, axis=0))

        ihvp = tf.stack(ihvp_list, axis=0)
        ihvp = tf.where(tf.math.is_finite(ihvp), ihvp, tf.zeros_like(ihvp))

        return tf.transpose(ihvp)



class IHVPCalculator(Enum):
    """
    Inverse Hessian Vector Product Calculator interface.
    """
    Exact = ExactIHVP
    Cgd = ConjugateGradientDescentIHVP
    Lissa = LissaIHVP
    Fisher = FisherIHVP
    KFAC = KFACIHVP
    EKFAC = EKFACIHVP

    @staticmethod
    def from_string(ihvp_calculator: str) -> 'IHVPCalculator':
        """
        Restore an IHVPCalculator from string.

        Parameters
        ----------
        ihvp_calculator
            String indicated the method use to compute the inverse hessian vector product,
            e.g 'exact', 'cgd', 'lissa', 'fisher', 'kfac', or 'ekfac'.

        Returns
        -------
        ivhp_calculator
            IHVPCalculator object.
        """
        valid_calculators = ['exact', 'cgd', 'lissa', 'fisher', 'kfac', 'ekfac']
        assert ihvp_calculator in valid_calculators, \
            f"Only {valid_calculators} inverse hessian vector product calculators are supported."

        if ihvp_calculator == 'exact':
            return IHVPCalculator.Exact
        if ihvp_calculator == 'lissa':
            return IHVPCalculator.Lissa
        if ihvp_calculator == 'fisher':
            return IHVPCalculator.Fisher
        if ihvp_calculator == 'kfac':
            return IHVPCalculator.KFAC
        if ihvp_calculator == 'ekfac':
            return IHVPCalculator.EKFAC

        return IHVPCalculator.Cgd
