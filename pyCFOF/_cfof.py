# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

import numpy as np
import warnings

from typing import List

from sklearn.neighbors._base import NeighborsBase
from sklearn.neighbors._base import KNeighborsMixin
from sklearn.base import OutlierMixin

from sklearn.utils.validation import check_is_fitted

__all__ = ["ConcentrationFreeOutlierFactor"]


class ConcentrationFreeOutlierFactor(NeighborsBase, KNeighborsMixin,
                                     OutlierMixin):
    """Unsupervised Outlier Detection using Concentration Free Outlier Factor (CFOF)

    The anomaly score of each sample is called Concentration Free Outlier Factor.

    Parameters
    ----------
    n_neighbors :
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.
        Must be defined, even if is changed after.

    rho : float, default=0.1
        Percentage of neighborhood.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to :class:`BallTree` or :class:`KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        metric used for the distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a sparse matrix, in which case only "nonzero"
        elements may be considered neighbors.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics:
        https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    p : int, default=2
        Parameter for the Minkowski metric from
        :func:`sklearn.metrics.pairwise.pairwise_distances`. When p = 1, this
        is equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    contamination : 'None' or float, default='None'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. When fitting this is used to define the
        threshold on the scores of the samples.

        - if 'None', return only the CFOF score
        - if a float, the contamination should be in the range [0, 0.5].

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Examples
    --------
    >>> from pyCFOF import ConcentrationFreeOutlierFactor as CFOF
    >>> X = [[-1.1], [0.2], [101.1], [0.3]]
    >>> cfof = CFOF(n_neighbors=len(X), rho=0.1)
    >>> cfof.fit_predict(X)
    array([[ 1],
           [ 1],
           [-1],
           [ 1]])
    >>> cfof.outlier_factor_
    array([[0.75],
           [0.5 ],
           [1.  ],
           [0.5 ]])

    References
    ----------
    .. [1] Angiulli, F. (2020, January).
           CFOF: A Concentration Free Measure for Anomaly Detection.
           In ACM Transactions on Knowledge Discovery from Data.
    """
    def __init__(self, n_neighbors: int = 20, rho: List[float] = None,
                 algorithm: str = 'auto', leaf_size: int = 30,
                 metric: str = 'minkowski', p=2, metric_params=None,
                 contamination: str = "auto", n_jobs=None):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size, metric=metric, p=p,
            metric_params=metric_params, n_jobs=n_jobs)
        if rho is None:
            rho = [0.1]
        try:
            rho = [x for x in rho if 0.0 < x < 1.0]
        except Exception as e:
            print(f'Rho type error: must be a float: {rho}.')
            print(f'Default value applied. {type(e)}: {e}')
            rho = [0.1]
        finally:
            if not rho:
                rho = [0.1]
        self.contamination = contamination
        self.rho = rho

    def _check_parameters(self, X):
        """
        Check if the initialization parameters are valid

        Returns
        -------
        self : object
        """
        self.rho = sorted(self.rho)

        if self.n_neighbors is None:
            self.n_neighbors = X.shape[0]

        return self

    @property
    def fit_predict(self):
        """"Fits the model to the training set X and returns the labels.

        Label is 1 for an inlier and -1 for an outlier according to the CFOF
        score and the contamination parameter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            The query sample or samples to compute the CFOF
            w.r.t. to the training samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and 1 for inliers.
        """

        return self._fit_predict

    def _fit_predict(self, X, y=None):
        """"Fits the model to the training set X and returns the labels.

        Label is 1 for an inlier and -1 for an outlier according to the CFOF
        score and the contamination parameter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            The query sample or samples to compute the CFOF
            w.r.t. to the training samples.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and 1 for inliers.
        """

        return self.fit(X)._predict()

    def fit(self, X, y=None):
        """Fit the model using X as training data.

        Parameters
        ----------
        X : BallTree, KDTree or {array-like, sparse matrix} of shape \
                (n_samples, n_features) or (n_samples, n_samples)
            Training data. If array or matrix, the shape is (n_samples,
            n_features), or (n_samples, n_samples) if metric='precomputed'.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
        """
        self._check_parameters(X)

        if self.contamination != 'auto':
            if not(0. < self.contamination <= .5):
                raise ValueError("contamination must be in (0, 0.5], "
                                 "got: %f" % self.contamination)

        self._fit(X)

        n_samples = self._fit_X.shape[0]
        if self.n_neighbors > n_samples:
            warnings.warn("n_neighbors (%s) is greater than the "
                          "total number of samples (%s). n_neighbors "
                          "will be set to (n_samples - 1) for estimation."
                          % (self.n_neighbors, n_samples))
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples-1))

        self._distances_fit_X_, _neighbors_indices_fit_X_ = self.kneighbors(
            n_neighbors=self.n_neighbors_)

        smallest_nn_width = self._smallest_neighborhood_width(_neighbors_indices_fit_X_, n_samples)

        self.outlier_factor_ = np.true_divide(smallest_nn_width, n_samples)

        # TODO adapt offset_ to multi rho values
        if self.contamination == "auto":
            # inliers score around 0
            self.offset_ = 0.75
        else:
            self.offset_ = np.percentile(self.outlier_factor_,
                                         100. * (1 - self.contamination))

        return self

    def _predict(self):
        """Predict the labels (1 inlier, -1 outlier) of X according to CFOF.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The query sample or samples to compute the CFOF.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and 1 for inliers.
        """
        check_is_fitted(self, ["offset_", "outlier_factor_",
                               "n_neighbors_", "_distances_fit_X_"])

        is_inlier = np.ones((self._fit_X.shape[0], len(self.rho)), dtype=int)
        is_inlier[self.outlier_factor_ > self.offset_] = -1

        return is_inlier

    def _smallest_neighborhood_width(self, neighbors_indices, n_samples):
        """
        For each query point, find the smallest neighborhood width for which
        it exhibits a reverse neighborhood of size at least
        self.n_neighbors_ * rho

        Parameters
        ----------
        neighbors_indices : ndarray of shape (n_queries, self.n_neighbors)
            Neighbors indices (of each query point) among training samples
            self._fit_X.

        n_samples : int, size of X.

        Returns
        -------
        smallest_neighborhood_width : ndarray of shape (n_queries, rho)
            The smallest neighborhood width of each sample.
        """

        k_tmp = np.ones(n_samples, dtype=np.dtype('uint32'))
        current_step = np.zeros(n_samples, dtype=np.dtype('uint32'))
        counter = np.zeros(n_samples, dtype=np.dtype('uint32'))
        score_list = np.ones((n_samples, len(self.rho)))

        logical_k_tmp = 1
        
        # used to not count two times same distance
        not_same_neighbor = (np.diff(np.concatenate((np.zeros_like(self._distances_fit_X_.T[0]).reshape(-1, 1),
                                                     self._distances_fit_X_), axis=1)) != 0).T
        not_same_neighbor = np.concatenate(([np.zeros_like(not_same_neighbor[0]).T], not_same_neighbor))

        k_tmp[not_same_neighbor[logical_k_tmp - 1]] += 1
        logical_k_tmp += 1

        rho_with_extra = self.rho + [1.1]

        for col_k in np.concatenate(([np.arange(neighbors_indices.shape[0], dtype=np.dtype('uint32'))], neighbors_indices.T)):

            counter = np.add(counter, np.bincount(col_k, minlength=n_samples))

            where_goal_rho = counter >= n_samples * np.take(rho_with_extra, current_step)

            score_list[where_goal_rho, current_step[where_goal_rho]] = k_tmp[where_goal_rho]

            while where_goal_rho.any():
                current_step = np.where(where_goal_rho,
                                        np.add(current_step, 1),
                                        current_step)
                where_goal_rho = counter >= n_samples * np.take(rho_with_extra, current_step)

            if np.all(current_step >= len(self.rho)):
                break

            not_same_neighbor[logical_k_tmp-1]
            k_tmp[not_same_neighbor[logical_k_tmp-1]] += 1
            logical_k_tmp += 1

        return score_list
