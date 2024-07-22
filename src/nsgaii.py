import pandas as pd
import os
import sys
from pathlib import Path
from warnings import simplefilter
import math

path_src = str(Path(os.getcwd(), "..").absolute())
sys.path.insert(0, "..")

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=RuntimeWarning)

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from xgboost import XGBClassifier


class MicroarrayProblem(ElementwiseProblem):
    def __init__(
        self,
        df,
        y,
        cv_k,
        n_max,
        evaluator_name="linearsvm",
    ):
        super().__init__(
            n_var=len(df.columns.to_numpy()), n_obj=2, n_ieq_constr=0, xl=0, xu=1
        )
        self.L = df.columns.to_numpy()
        self.n_max = n_max
        self.df = df
        self.X_e = df.to_numpy()
        self.y_e = y.to_numpy()
        self.cv_k = cv_k
        self.evaluator_name = evaluator_name

    def _evaluate(self, x, out, *args, **kwargs):
        # Create an SVM classif.
        if self.evaluator_name == "linearsvm":
            clf = LinearSVC(max_iter=3000, dual=True)  # , random_state=42)
        if self.evaluator_name == "knn":
            clf = KNeighborsClassifier(n_neighbors=5)
        if self.evaluator_name == "decisiontree":
            clf = DecisionTreeClassifier()
        if self.evaluator_name == "xgb":
            clf = XGBClassifier()
        cv = StratifiedKFold(n_splits=self.cv_k, shuffle=True)
        X_e = self.X_e[:, x]
        y_e = self.y_e

        scores = cross_validate(
            clf,
            X_e,
            y_e.ravel(),
            cv=cv,
            scoring=[
                "f1_micro",
                "f1_macro",
                "f1_weighted",
                "recall_micro",
                "recall_macro",
                "recall_weighted",
                "accuracy",
            ],
            return_estimator=False,
            n_jobs=-1,
        )

        out["F"] = [1 - np.mean(scores["test_f1_macro"]), X_e.shape[1]]


from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[: problem.n_max]
            X[k, I] = True

        return X


class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True
        print(f"Generated {len(_X)} new samples through crossover.")

        return _X


class NoMutation(Mutation):
    def __do(self, problem, X, **kwargs):
        return X

    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false)] = True
            X[i, np.random.choice(is_true)] = False
        print(f"Mutated {len(X)} individuals.")

        return X


def get_feature_set_mask(values, n_features):
    # Get first n_features

    indexes = np.full(values.shape, False, dtype=bool)
    indexes[0:n_features] = True
    return indexes


def get_feature_mask_by_importance(
    values,
    n_samples,
    n_features,
    max_attempts=100,
    prob_multiplier=1.25,
    min_features=None,
):
    # Sort a value for each feature between df.min and df.max (from values).
    # Then select the top n_features that surpass this sorted value.
    # Adds some randomness while favoring top features.

    if min_features:
        size_constraint = min_features
    else:
        size_constraint = n_features

    if n_features >= values.shape[0]:
        raise Exception("Not enough features (n_features >= len(values))")

    selected_indexes = []
    selected_index_masks = []
    for sample_n in range(n_samples):
        selected_index = []
        attempts = 0

        # While selected_index is empty or another identical selected_index was already selected
        while len(selected_index) == 0 or np.any(
            [np.all(selected_index == item) for item in selected_indexes]
        ):
            selected_index = []
            while len(selected_index) < size_constraint:
                values_ = values.copy()
                # Set values for features already selected
                values_[selected_index] = np.min(values_)
                # Randomize values to select new candidate features (not selected yet)
                chances = np.random.uniform(
                    low=np.min(values_),
                    high=np.max(values_) * prob_multiplier,
                    size=values_.shape,
                )
                # Set chances for features already selected
                chances[selected_index] = np.min(values_)
                # Verify chosen features
                results = [
                    True if x >= y else False for (x, y) in zip(values_, chances)
                ]
                # Select top n_features
                selected_index = np.argwhere(results)[0:n_features, 0]

                # if min_features is set, select random final_n_features between min_features and n_features
                if min_features and len(selected_index) >= size_constraint:
                    upper_limit = min(n_features, len(selected_index))
                    final_n_features = (
                        np.random.randint(min_features, upper_limit)
                        if upper_limit > min_features
                        else min_features
                    )
                    selected_index = sorted(
                        np.random.choice(
                            selected_index, final_n_features, replace=False
                        )
                    )

                # if the feature set already exists in the collection, try flipping a bit to force diversion
                while np.any(
                    [np.all(selected_index == item) for item in selected_indexes]
                ):
                    # if min_features is set
                    if min_features and len(selected_index) >= size_constraint:
                        upper_limit = min(n_features, len(selected_index))
                        final_n_features = (
                            np.random.randint(min_features, upper_limit)
                            if upper_limit > min_features
                            else min_features
                        )
                        selected_index = sorted(
                            np.random.choice(
                                selected_index, final_n_features, replace=False
                            )
                        )
                    else:
                        final_n_features = n_features

                    # if the selected_index is present in the collection, try flipping a bit to force diversion
                    if (
                        np.any(
                            [
                                np.all(selected_index == item)
                                for item in selected_indexes
                            ]
                        )
                        and len(selected_index) >= size_constraint
                    ):
                        flip_position = np.random.choice(selected_index)
                        results[flip_position] = not results[flip_position]
                        selected_index = np.argwhere(results)[0:final_n_features, 0]

                    # Limit attempts to a maximum amount. Infinite loops are not yet defined.
                    attempts += 1
                    if attempts >= max_attempts:
                        raise Exception(
                            f"Attempts to generate new feature sets reached max_attempts ({max_attempts}). Amount of sets created: {len(selected_index_masks)}"
                        )

        top_n_selected_index = selected_index
        selected_indexes.append(top_n_selected_index)
        index_mask = np.full(values.shape[0], False, dtype=bool)
        index_mask[top_n_selected_index] = True
        selected_index_masks.append(index_mask)

    return np.array(selected_index_masks)


def _get_feature_mask_by_importance(
    values,
    n_samples,
    n_features,
    max_attempts=100,
    prob_multiplier=1.25,
    min_features=None,
):
    # Sort a value for each feature between df.min and df.max (from values).
    # Then select the top n_features that surpass this sorted value.
    # Adds some randomness while favoring top features.

    if min_features:
        min_size_constraint = min_features
    else:
        min_size_constraint = n_features

    if n_features >= values.shape[0]:
        raise Exception("Not enough features (n_features >= len(values))")

    selected_indexes = []
    selected_index_masks = []
    feature_n_distrib = np.random.randint(
        min_size_constraint, n_features + 1, size=n_samples
    )
    print(f"Generating {n_samples} feature sets with distribution: {feature_n_distrib}")
    for sample_n in range(n_samples):
        selected_index = []
        attempts = 0
        final_n_features = feature_n_distrib[sample_n]

        # While selected_index is empty or another identical selected_index was already selected
        while len(selected_index) == 0 or np.any(
            [np.all(selected_index == item) for item in selected_indexes]
        ):
            selected_index = []
            while len(selected_index) < min_size_constraint:
                values_ = values.copy()
                # Set values for features already selected
                values_[selected_index] = np.min(values_)
                # Randomize values to select new candidate features (not selected yet)
                chances = np.random.uniform(
                    low=np.min(values_),
                    high=np.max(values_) * prob_multiplier,
                    size=values_.shape,
                )
                # Set chances for features already selected
                chances[selected_index] = np.min(values_)
                # Verify chosen features
                results = [
                    True if x >= y else False for (x, y) in zip(values_, chances)
                ]
                # Select top n_features
                selected_index = np.argwhere(results)[0:n_features, 0]

                # if min_features is set, select random final_n_features between min_features and n_features
                if min_features and len(selected_index) >= min_size_constraint:
                    final_n_features = min(final_n_features, len(selected_index))
                    selected_index = sorted(
                        np.random.choice(
                            selected_index, final_n_features, replace=False
                        )
                    )

                # if the feature set already exists in the collection, try flipping a bit to force diversion
                while np.any(
                    [np.all(selected_index == item) for item in selected_indexes]
                ):
                    # if min_features is set
                    if min_features and len(selected_index) >= min_size_constraint:
                        final_n_features = min(final_n_features, len(selected_index))
                        selected_index = sorted(
                            np.random.choice(
                                selected_index, final_n_features, replace=False
                            )
                        )
                    else:
                        final_n_features = n_features

                    # if the selected_index is present in the collection, try flipping a bit to force diversion
                    if (
                        np.any(
                            [
                                np.all(selected_index == item)
                                for item in selected_indexes
                            ]
                        )
                        and len(selected_index) >= min_size_constraint
                    ):
                        flip_position = np.random.choice(selected_index)
                        results[flip_position] = not results[flip_position]
                        selected_index = np.argwhere(results)[0:final_n_features, 0]

                    # Limit attempts to a maximum amount. Infinite loops are not yet defined.
                    attempts += 1
                    if attempts >= max_attempts:
                        raise Exception(
                            f"Attempts to generate new feature sets reached max_attempts ({max_attempts}). Amount of sets created: {len(selected_index_masks)}"
                        )

        top_n_selected_index = selected_index
        selected_indexes.append(top_n_selected_index)
        index_mask = np.full(values.shape[0], False, dtype=bool)
        index_mask[top_n_selected_index] = True
        selected_index_masks.append(index_mask)

    print(
        f"Generated feature sets with distribution: {[np.sum(mask) for mask in selected_index_masks]}"
    )

    return np.array(selected_index_masks)


def get_feature_mask_by_rank(values, n_samples, max_features, min_features=1):
    """
    Generates n_samples features masks starting with top min_features and, for each next sample until n_samples are created, adding the next subsequent feature.
    """

    print(f"Generating {n_samples} feature sets.")

    if n_samples > len(range(min_features, max_features + 1)):
        raise Exception(
            "Not enough features to complete the sample set (n_samples > diff(min_features, max_features))"
        )

    selected_index_masks = []
    for sample_n in range(min_features, n_samples + 1):
        index_mask = np.full(values.shape[0], False, dtype=bool)
        top_n_selected_index = range(0, sample_n)
        index_mask[top_n_selected_index] = True
        selected_index_masks.append(index_mask)

    print(
        f"Generated feature sets with distribution: {[np.sum(mask) for mask in selected_index_masks]}"
    )

    return np.array(selected_index_masks)


class FeatureSampling(Sampling):
    def __init__(self, max_attempts=100, prob_multiplier=1.25, min_features=2):
        self.max_attempts = max_attempts
        self.prob_multiplier = prob_multiplier
        self.min_features = min_features
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        # Load or generate feature importances. I think we could generate them before and just load them here.
        # Based on importances, we can set a chance of each feature being selected. Then generate feature sets randomly based on them.
        # We can specify which portion of n_samples will be generated by each method

        """
        feature_dfs = {...}
        fs_distrib = {"mrmr": 0.2, "relieff": 0.2, "kruskalwallis": 0.2, "mutualinfo": 0.2, "decisiontree": 0.2}
        """
        # feature_dfs = kwargs["feature_dfs"] Apparently kwargs doesn`t reach this point.... -_-
        # fs_distrib = kwargs["fs_distrib"]

        feature_dfs = problem.feature_dfs
        fs_distrib = problem.fs_distrib
        methods = [method for method in fs_distrib]

        X = []
        for method in methods:
            feature_df, fs_proportion = feature_dfs[method], fs_distrib[method]

            X_ = get_feature_mask_by_importance(
                values=feature_df.values,
                n_samples=int(n_samples * fs_proportion),
                n_features=problem.n_max,
                max_attempts=self.max_attempts,
                prob_multiplier=self.prob_multiplier,
                min_features=self.min_features,
            )

            # Reorder the resulting array - the feature_df does not use the same sorting as the original problem.df
            idx1 = np.argsort(problem.df.columns)
            idx2 = np.argsort(feature_df.index)
            idx1_inv = np.argsort(idx1)
            X_ = X_[:, idx2][:, idx1_inv]

            X.append(X_)

        X = np.concatenate(X)
        print(f"Generated {len(X)} new samples.")

        return X


class StrictFeatureSampling(Sampling):
    def __init__(self, best_fs_set, min_features=2):
        self.best_fs_set = best_fs_set
        self.min_features = min_features
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        # Load or generate feature importances. I think we could generate them before and just load them here.
        # Based on importances, we can set a chance of each feature being selected. Then generate feature sets randomly based on them.
        # We can specify which portion of n_samples will be generated by each method

        """
        feature_dfs = {...}
        fs_distrib = {"mrmr": 0.2, "relieff": 0.2, "kruskalwallis": 0.2, "mutualinfo": 0.2, "decisiontree": 0.2}
        """

        feature_dfs = problem.feature_dfs
        fs_distrib = problem.fs_distrib
        methods = [method for method in fs_distrib]

        # Get first n_max feature sets from the best method
        X = []
        n_samples_ = min(n_samples, problem.n_max - self.min_features)
        feature_df = feature_dfs[self.best_fs_set]
        X_ = get_feature_mask_by_rank(
            values=feature_df.values,
            n_samples=n_samples_,
            max_features=problem.n_max,
            min_features=self.min_features,
        )

        # Reorder the resulting array - the feature_df does not use the same sorting as the original problem.df
        idx1 = np.argsort(problem.df.columns)
        idx2 = np.argsort(feature_df.index)
        idx1_inv = np.argsort(idx1)
        X_ = X_[:, idx2][:, idx1_inv]
        X.append(X_)

        # Get remaining features from randomized masks based on fs datasets
        n_samples_remaining = max(n_samples - n_samples_, 0)
        X_ = []
        if n_samples_remaining > 0:
            for method in methods:
                feature_df, fs_proportion = feature_dfs[method], fs_distrib[method]

                X__ = get_feature_mask_by_importance(
                    values=feature_df.values,
                    n_samples=math.ceil(n_samples_remaining * fs_proportion),
                    n_features=problem.n_max,
                    max_attempts=self.max_attempts,
                    prob_multiplier=self.prob_multiplier,
                    min_features=self.min_features,
                )

                # Reorder the resulting array - the feature_df does not use the same sorting as the original problem.df
                idx1 = np.argsort(problem.df.columns)
                idx2 = np.argsort(feature_df.index)
                idx1_inv = np.argsort(idx1)
                X__ = X__[:, idx2][:, idx1_inv]
                X_.append(X__)

        X.append(X_[0:n_samples_remaining])
        X = np.concatenate(X)

        print(f"Generated {len(X)} new samples.")

        return X


class FeatureSamplingMutation(Mutation):
    def __do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false)] = True
            X[i, np.random.choice(is_true)] = False
        print(f"Mutated {len(X)} individuals.")

        return X

    def _do(self, problem, X, **kwargs):
        # Load or generate feature importances. I think we could generate them before and just load them here.
        # Based on importances, we can set a chance of each feature being selected. Then generate feature sets randomly based on them.
        # We can specify which portion of n_samples will be generated by each method

        """
        feature_dfs = {...}
        fs_distrib = {"mrmr": 0.2, "relieff": 0.2, "kruskalwallis": 0.2, "mutualinfo": 0.2, "decisiontree": 0.2}
        """

        feature_dfs = problem.feature_dfs
        fs_distrib = problem.fs_distrib
        methods = [method for method in fs_distrib]

        feature_df = feature_dfs[np.random.choice(methods)]

        # Gets a subset of features of size (n_max) to use as a source of features
        feature_mask = get_feature_mask_by_importance(
            values=feature_df.values,
            n_samples=1,
            n_features=problem.n_max,
            max_attempts=100,
            prob_multiplier=1,
            min_features=problem.n_max,
        )

        # Reorder the resulting array - the feature_df does not use the same sorting as the original problem.df
        idx1 = np.argsort(problem.df.columns)
        idx2 = np.argsort(feature_df.index)
        idx1_inv = np.argsort(idx1)
        feature_mask = feature_mask[:, idx2][:, idx1_inv]

        mask_true = np.where(feature_mask)[1]

        # Iterates over the population performing mutations
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_true = np.where(X[i, :])[0]
            is_false = np.where(np.logical_not(X[i, :]))[0]

            available_features = np.intersect1d(mask_true, is_false)

            if len(available_features) > 0:
                # Enables one feature
                X[i, np.random.choice(available_features)] = True
                # Disables one feature
                X[i, np.random.choice(is_true)] = False
        print(f"Mutated {len(X)} individuals with feature sampling.")

        return X
