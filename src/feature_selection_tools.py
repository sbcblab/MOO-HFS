import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.commons import *


def generate_n_importances(X_e, y_e, feature_names, dst_path, n, method_type="all"):
    if method_type == "all":
        methods_ = methods
    elif method_type == "deterministic":
        methods_ = deterministic_methods
    elif method_type == "non_deterministic":
        methods_ = non_deterministic_methods

    importances_dict = {}
    for method in methods_:
        # First pass of selections
        importances = method(X_e, y_e, feature_names, dst_path)

        # Reduce the original dataset to a smaller version with only n features
        X_selected, top_feature_names = select_top_n(X_e, feature_names, importances, n)

        # Perform feature selection again
        selected_importances = method(X_selected, y_e, top_feature_names, dst_path)

        # Set the discarded features to zero importance and recreate the series
        discarded_importances = importances[~importances.isin(selected_importances)]
        discarded_importances.index = np.zeros(len(discarded_importances))
        final_importances = pd.concat(
            [selected_importances, discarded_importances], axis=0
        )

        save_importances(
            final_importances.values,
            final_importances.index.values,
            method.__name__,
            dst_path,
            ascending=False,
        )
        importances_dict[method.__name__] = final_importances

    return importances_dict


def generate_n_degrees_importances(X_e, y_e, feature_names, dst_path, n, degrees):
    for method in methods:
        # First pass of selections
        importances = method(X_e, y_e, feature_names)

        X_selected = X_e
        top_feature_names = feature_names
        selected_importances = importances

        for n in degrees:
            # Reduce the original dataset to a smaller version with only n features
            X_selected, top_feature_names = select_top_n(
                X_selected, top_feature_names, selected_importances, n
            )

            # Perform feature selection again
            selected_importances = method(X_selected, y_e, top_feature_names)

        # Set the discarded features to zero importance and recreate the series
        discarded_importances = importances[~importances.isin(selected_importances)]
        discarded_importances.index = np.zeros(len(discarded_importances))
        final_importances = pd.concat(
            [selected_importances, discarded_importances], axis=0
        )

        save_importances(
            final_importances.values,
            final_importances.index.values,
            method.__name__,
            dst_path,
            ascending=False,
        )


def generate_importances(X_e, y_e, feature_names, dst_path, method_type="all"):
    """_summary_

    Args:
        X_e (_type_): Input feature dataset. Ideally, encoded.
        y_e (_type_): Input label dataset. Ideally, encoded.
        feature_names (_type_): List of feature names for the input feature dataset.
        dst_path (_type_): Path for the output files.
        method_type (str, optional): Types of methods to use. "all", "deterministic" or "non_deterministic". Defaults to "all".

    Returns:
        _type_: _description_
    """

    if method_type == "all":
        methods_ = methods
    elif method_type == "deterministic":
        methods_ = deterministic_methods
    elif method_type == "non_deterministic":
        methods_ = non_deterministic_methods

    for method in methods_:
        method(X_e, y_e, feature_names, dst_path)


def save_importances(feature_names, importances, method, dst_path, ascending=False):
    # Save importances to a .csv file following the wTSNE format

    series = pd.Series(feature_names, index=importances).sort_index(ascending=ascending)
    logging.debug(series)

    logging.info(
        f"Saving importances for {method} method with {len(feature_names)} features."
    )
    pd.DataFrame(series, columns=["feature"]).reset_index().rename(
        columns={"index": "value"}
    )[["feature", "value"]].to_csv(
        f"{dst_path}/{method}_{len(feature_names)}_importances.csv", index=False
    )

    return series


def select_top_n(X, feature_names, importances, n):
    # Select top feature from the numpy array version of a dataset

    top_feature_names = importances.iloc[:n].values
    top_feature_indexes = np.array(
        [np.argwhere(feature_names == x) for x in top_feature_names]
    ).ravel()
    X_selected = X[:, top_feature_indexes]

    return X_selected, top_feature_names


def generate_relieff_importances(X_e, y_e, feature_names, dst_path):
    from skrebate import ReliefF

    method = "relieff"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    reliefF_clf = ReliefF(n_features_to_select=15, n_neighbors=100, n_jobs=-1).fit(
        X_e, y_e.ravel()
    )
    importances = reliefF_clf.feature_importances_
    sorted_importances = save_importances(
        feature_names, importances, method, dst_path, ascending=False
    )

    return sorted_importances


def generate_kruskalwallis_importances(X_e, y_e, feature_names, dst_path):
    from scipy.stats import kruskal

    method = "kruskalwallis"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    label_groups = (X_e[np.argwhere(y_e == label)[:, 0]] for label in np.unique(y_e))
    res = kruskal(*label_groups)
    importances = (
        res.statistic
    )  # statistic, pvaluer for p-value (the closest to zero the the better)
    sorted_importances = save_importances(
        feature_names, importances, method, dst_path, ascending=False
    )

    return sorted_importances


def generate_mutualinfo_importances(X_e, y_e, feature_names, dst_path):
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    method = "mutualinfo"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    importances = mutual_info_classif(
        X_e,
        y_e.ravel(),
        n_neighbors=3,
    )
    # !!! Documentation is a little confusing, but assuming higher importances = more representative = better features (also checks with tests made on data).
    sorted_importances = save_importances(
        feature_names, importances, method, dst_path, ascending=False
    )

    return sorted_importances


def generate_lassocv_importances(X_e, y_e, feature_names, dst_path):
    from sklearn.linear_model import LassoCV

    method = "lassocv"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    clf = LassoCV().fit(X_e, y_e.ravel())
    importances = np.abs(clf.coef_)
    sorted_importances = save_importances(
        feature_names, importances, method, dst_path, ascending=False
    )

    return sorted_importances


def generate_decisiontree_importances(X_e, y_e, feature_names, dst_path):
    from sklearn.tree import DecisionTreeClassifier

    method = "decisiontree"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    clf = DecisionTreeClassifier().fit(X_e, y_e.ravel())
    importances = clf.feature_importances_
    sorted_importances = save_importances(
        feature_names, importances, method, dst_path, ascending=False
    )

    return sorted_importances


def generate_randomforest_importances(
    X_e, y_e, feature_names, dst_path, gridsearch=False
):
    from sklearn.ensemble import RandomForestClassifier

    method = "randomforest"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    if gridsearch:
        logging.info("Using GridSearch.")
        param_grid = {
            "n_estimators": [100, 150],
            "max_features": ["sqrt", "log2"],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 3],
        }
        grid_cv = GridSearchCV(RandomForestClassifier(), param_grid).fit(
            X_e, y_e.ravel()
        )
        logging.info(f"Params chosen by GridSearch: {grid_cv.best_params_}")

        clf = grid_cv.best_estimator_
    else:
        # Using a larger number of estimators by standard (default is 100)
        clf = RandomForestClassifier(
            n_estimators=150,
        ).fit(X_e, y_e.ravel())

    importances = clf.feature_importances_
    sorted_importances = save_importances(
        feature_names, importances, method, dst_path, ascending=False
    )

    return sorted_importances


def generate_linearsvm_importances(X_e, y_e, feature_names, dst_path, gridsearch=False):
    from sklearn.svm import LinearSVC

    method = "linearsvm"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    if gridsearch:
        logging.info("Using GridSearch.")
        param_grid = {
            "tol": [1e-6, 1e-4, 1e-2],
            "C": [1, 10],
            "max_iter": [1000, 2000],
            "dual": [True],
        }
        grid_cv = GridSearchCV(LinearSVC(), param_grid).fit(X_e, y_e.ravel())
        logging.info(f"Params chosen by GridSearch: {grid_cv.best_params_}")

        clf = grid_cv.best_estimator_
    else:
        clf = LinearSVC(dual=True).fit(X_e, y_e.ravel())

    svm_weights = np.abs(clf.coef_).sum(axis=0)
    svm_weights /= svm_weights.sum()
    importances = svm_weights
    sorted_importances = save_importances(
        feature_names, importances, method, dst_path, ascending=False
    )

    return sorted_importances


def generate_anovafvalue_importances(X_e, y_e, feature_names, dst_path):
    from sklearn.feature_selection import f_classif

    method = "anovafvalue"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    clf = f_classif(X_e, y_e.ravel())
    importances = clf[
        0
    ]  # [0] for statistic (the higher, the better), [1] for p-value (the closest to zero the the better)
    sorted_importances = save_importances(
        feature_names, importances, method, dst_path, ascending=False
    )

    return sorted_importances


def generate_mrmr_importances(X_e, y_e, feature_names, dst_path, k=100):
    from mrmr import mrmr_classif

    method = "mrmr"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")

    selected_features = mrmr_classif(
        X=pd.DataFrame(X_e, columns=feature_names),
        y=pd.Series(y_e.ravel()),
        K=k,
        return_scores=True,
    )
    top_k_features = selected_features[0]
    scores = list(range(len(top_k_features), 0, -1))
    print(selected_features)

    importances = np.zeros(len(feature_names))
    feature_pos = np.array(
        [np.where(feature_names == x) for x in top_k_features]
    ).ravel()
    importances[feature_pos] = scores

    sorted_importances = save_importances(
        feature_names, importances, method, dst_path, ascending=False
    )

    return sorted_importances


deterministic_methods = [
    # Instant datasets (<60s)
    generate_kruskalwallis_importances,
    generate_anovafvalue_importances,
    # A few more minutes (>10min)
    generate_lassocv_importances,
    # A lot of minutes (>60min, +-3h for +-60k features)
    generate_relieff_importances,
    generate_mrmr_importances,
]

non_deterministic_methods = [
    # Instant datasets (<60s)
    generate_decisiontree_importances,
    generate_randomforest_importances,
    generate_linearsvm_importances,
    # A few minutes (<10min)
    generate_mutualinfo_importances,
]

methods = deterministic_methods + non_deterministic_methods
