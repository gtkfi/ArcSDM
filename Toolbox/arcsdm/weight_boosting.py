
"""Weight Boosting
This module contains weight boosting estimators for both classification and
regression.
The module structure is the following:
- The ``BaseWeightBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ from each other in the loss function that is optimized.
- ``AdaBoostClassifier`` implements adaptive boosting (AdaBoost-SAMME) for
  classification problems.
- ``AdaBoostRegressor`` implements adaptive boosting (AdaBoost.R2) for
  regression problems.
"""

# Authors: Noel Dawe <noel@dawe.me>
#          Gilles Louppe <g.louppe@gmail.com>
#          Hamzeh Alsalhi <ha258@cornell.edu>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#
# License: BSD 3 clause
# TODO: Redo documentation
# TODO: Clean up object

import numpy as np
from numpy.core.umath_tests import inner1d

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.ensemble.weight_boosting import BaseWeightBoosting

__all__ = [
    'BrownBoostClassifier',
]


def _samme_proba(estimator, n_classes, X):
    """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].
    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
    """
    proba = estimator.predict_proba(X)

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
    log_proba = np.log(proba)

    return (n_classes - 1) * (log_proba - (1. / n_classes)
                              * log_proba.sum(axis=1)[:, np.newaxis])


def solve_de(r, hypothesis, response, s, c, v=0.001):
    hy = np.multiply(hypothesis, response)
    def dif_eq_const(a, t):
        return diff_eq(a,t, hy, r, s, c)
    h = s
    a = t = 0
    for i in xrange(1000):
        val_t, val_t2 = rk4_step(dif_eq_const,a,t,h)
        if abs(val_t - val_t2) > v:
            h /= 2
            continue
        val_a = a + h
        gamma = dif_eq_const(val_a,val_t)
        if val_t >= s:
            return a + h*(s-t)/(val_t-t), s
        if gamma < v:
            if gamma < 0:
                h /= 2
                continue
            return val_a, val_t
        a = val_a
        t = val_t

    raise ArithmeticError("Solution of differential equation not found")

def rk4_step(f, x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(x + h, y + k3)
    vy = y + 1. * (k1 + k2 + k2 + k3 + k3 + k4) / 6
    vy2 = y + 1. * k1
    return  vy, vy2

def diff_eq(a, t, hy, r, s, c):
    exponential = np.exp(-((r + a * hy + s - t) ** 2) / c)
    numerator = sum(np.multiply(exponential, hy))
    denominator = sum(exponential)
    return numerator / denominator


class BrownBoostClassifier(BaseWeightBoosting, ClassifierMixin):
    """An AdaBoost classifier.
    An AdaBoost [1] classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.
    This class implements the algorithm known as AdaBoost-SAMME [2].
    Read more in the :ref:`User Guide <adaboost>`.
    Parameters
    ----------
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.
    classes_ : array of shape = [n_classes]
        The classes labels.
    n_classes_ : int
        The number of classes.
    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.
    estimator_errors_ : array of floats
        Classification error for each estimator in the boosted
        ensemble.
    feature_importances_ : array of shape = [n_features]
        The feature importances if supported by the ``base_estimator``.
    See also
    --------
    AdaBoostRegressor, GradientBoostingClassifier, DecisionTreeClassifier
    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.
    .. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 countdown=10.,
                 random_state=None
                 ):

        BaseWeightBoosting.__init__(self,
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.countdown = 1. * countdown
        self.algorithm = algorithm

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        y : array-like of shape = [n_samples]
            The target values (class labels).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that algorithm is supported
        if self.algorithm not in ('SAMME', 'SAMME.R', 'BROWNIAN'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        if self.algorithm == 'BROWNIAN':
            # The initial prediction values of all examples is zero r1(x,y)=0.
            # Initialize remaining time s1=c.
            self.prediction_values = np.zeros_like(y)
            self.remaining_time = self.countdown

        # Fit
        return BaseWeightBoosting.fit(self, X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        BaseWeightBoosting._validate_estimator(self,
            default=DecisionTreeClassifier(max_depth=1))

        #  SAMME-R requires predict_proba-enabled base estimators
        if self.algorithm == 'SAMME.R':
            if not hasattr(self.base_estimator_, 'predict_proba'):
                raise TypeError(
                    "AdaBoostClassifier with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead.")
        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError("%s doesn't support sample_weight."
                             % self.base_estimator_.__class__.__name__)

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.
        Perform a single boost according to the real multi-class SAMME.R
        algorithm or to the discrete SAMME algorithm and return the updated
        sample weights.
        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        y : array-like of shape = [n_samples]
            The target values (class labels).
        sample_weight : array-like of shape = [n_samples]
            The current sample weights.
        random_state : numpy.RandomState
            The current random number generator
        Returns
        -------
        sample_weight : array-like of shape = [n_samples] or None
            The reweighted sample weights.
            If None then boosting has terminated early.
        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.
        estimator_error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        if self.algorithm == 'SAMME.R':
            return self._boost_real(iboost, X, y, sample_weight, random_state)

        elif self.algorithm == "SAMME":
            return self._boost_discrete(iboost, X, y, sample_weight,
                                        random_state)
        elif self.algorithm == "BROWNIAN":
            return self._boost_brownian(iboost, X, y, sample_weight, random_state)

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * self.learning_rate
                                * (((n_classes - 1.) / n_classes) *
                                   inner1d(y_coding, np.log(y_predict_proba))))

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, 1., estimator_error

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, estimator_weight, estimator_error

    def _boost_brownian(self, iboost, X, y, sample_weight, random_state):

        s = self.remaining_time
        if s <= 0:
            return None, None, None

        #   2. Call WeakLearn with the distribution defined by normalizing Wi(x;y) and receive from it a hypothesis
        #       hi(x)which has some advantage over random guessing
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        incorrect = y_predict != y
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in BrownBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        c = self.countdown
        r = self.prediction_values
        # 3. Let \gamma and t be real valued variables that obey the following differential equation:
        #           dt                 SUM _(x,y in T) e ^ (-(r_i(x,y)+\alpha h_i(x) y + s_i - t)^2/c) h_i(x) y
        #       ---------- = \gamma = --------------------------------------------------------------------------
        #        d \alpha                  SUM _(x,y in T) e ^ (-(r_i(x,y)+\alpha h_i(x) y + s_i - t)^2/c)
        #       Where r_i(x,y); h_i(x)y and s_i are all constants in this context.
        #       Given the boundary conditions t=0,\alpha=0 solve the set of equations to find t_i=t*>0 and
        #           \alpha _i= \alpha*   such that either \gamma* <= v or t*=s_i.
        estimator_weight, t = solve_de(r, y_predict, y, s, c)


        #   4. Update the prediction value of each example to r_(i+1)(x,y)=r_i(x,y)+\alpha _i h_i(x)y
        self.prediction_values += estimator_weight * np.multiply(y_predict, y)
        #   5. update ?remaining time? s_i+1=s_i - t_i
        self.remaining_time -= t

        if iboost == self.n_estimators - 1 and self.remaining_time >= 0:
            raise ValueError("Brownboost did {} rounds without finalizing, consider using a smaller countdown value (Remaining time: {})".format(self.n_estimators,self.remaining_time))
        sample_weight = np.exp(-(r + s) ** 2 / c)



        return sample_weight, estimator_weight, estimator_error


    def predict(self, X):
        """Predict classes for X.
        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def staged_predict(self, X):
        """Return staged predictions for X.
        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.
        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : generator of array, shape = [n_samples]
            The predicted classes.
        """
        n_classes = self.n_classes_
        classes = self.classes_

        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))

        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(
                    np.argmax(pred, axis=1), axis=0))

    def decision_function(self, X):
        """Compute the decision function of ``X``.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(_samme_proba(estimator, n_classes, X)
                       for estimator in self.estimators_)
        else:   # self.algorithm == "SAMME"
            pred = sum((estimator.predict(X) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.
        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_pred = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_pred = estimator.predict(X)
                current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm

    def predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        check_is_fitted(self, "n_classes_")

        n_classes = self.n_classes_
        X = self._validate_X_predict(X)

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            proba = sum(_samme_proba(estimator, n_classes, X)
                        for estimator in self.estimators_)
        else:   # self.algorithm == "SAMME" or self.algorithm == "BRAWNIAN"
            proba = sum(estimator.predict_proba(X) * w
                        for estimator, w in zip(self.estimators_,
                                                self.estimator_weights_))

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba

    def staged_predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.
        This generator method yields the ensemble predicted class probabilities
        after each iteration of boosting and therefore allows monitoring, such
        as to determine the predicted class probabilities on a test set after
        each boost.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        p : generator of array, shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        proba = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_proba = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_proba = estimator.predict_proba(X) * weight

            if proba is None:
                proba = current_proba
            else:
                proba += current_proba

            real_proba = np.exp((1. / (n_classes - 1)) * (proba / norm))
            normalizer = real_proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            real_proba /= normalizer

            yield real_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.
        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        return np.log(self.predict_proba(X))

