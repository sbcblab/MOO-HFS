import os
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd

import src.feature_generation_tools as generation
import src.feature_selection_tools as selection
from src.commons import *


class FeatureGenerationPipeline(ABC):
    def __init__(self, experiment_id=None) -> None:
        self.source_data_file = f"""{self.base_dir}/{self.source_data_name}.csv"""
        if not experiment_id:
            experiment_id = datetime.now().strftime("%m-%d-%Y--%H-%M-%S")
        self.experiment_id = experiment_id
        self.results_data_dir = f"""{self.results_dir}/{experiment_id}"""
        self.results_data_prefix = (
            f"""{self.results_data_dir}/{self.source_data_name}"""
        )

        if not os.path.exists(self.results_data_dir):
            os.makedirs(self.results_data_dir)

        setup_logging(f"""{self.results_data_dir}/execution.log""")

        super().__init__()

    @abstractmethod
    def transform(self):
        pass

    def get_source_dfs(self):
        source_df = pd.read_csv(self.source_data_file)
        y_source_df = (
            source_df[[self.y_col_name]]
            .rename_axis(self.index_col_name)
            .rename(columns={self.y_col_name: "y"})
        )

        return source_df, y_source_df

    def split_source_df(self, source_df):
        # Transform source dfs into a splitted state (column values are exploded into multiple columns)
        target_cols = list(source_df.columns)
        target_cols.remove(self.y_col_name)
        X_splitted = X_splitted[target_cols].rename_axis(self.index_col_name)

        return X_splitted

    def prepare_dfs(self, X, y, save=True):
        if self.y_col_name in X.columns:
            X = X.drop(self.y_col_name, axis=1)

        X_e, y_e, feature_names = generation.prepare_dataset(X, y)
        not_nan_indexes = ~np.isnan(X_e).any(axis=1)

        # Some datasets have NaN values in some rows and this invalidates some of the methods tested.
        # We are choosing to remove the rows here instead of removing the columns, but the other way is also possible
        X_e = X_e[not_nan_indexes]
        y_e = y_e[not_nan_indexes]

        if save:
            encoded_df = pd.DataFrame(X_e, columns=feature_names, index=X.index)
            encoded_df[self.y_col_name] = y
            encoded_df.to_csv(
                f"""{self.base_dir}/{self.source_data_name}_encoded.csv"""
            )

        return X_e, y_e, feature_names

    def generate_feature_dfs(
        self, source_df, y_source_df, runs=1, select=-1, enable_repeated=False
    ):
        """_summary_

        Args:
            source_df (_type_): source pandas DF to perform selection, with no labels.
            y_source_df (_type_): labels for source DF.
            runs (int, optional): Number of runs to generate. Defaults to 1.
            select (int, optional): Number of features to generate weights.
                -1 for all features, (0, inf) for top <select> features.
                Remaining features will receive weight 0. Defaults to -1.
        """

        X_e, y_e, feature_names = self.prepare_dfs(X=source_df, y=y_source_df)

        for run in range(0, runs):
            dst_path = f"{self.results_data_dir}/run_{run}/importances/"

            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            if select == -1:
                if run == 0 or enable_repeated:
                    logging.info(f"Running all methods for run {run+1}/{runs}.")
                    selection.generate_importances(
                        X_e, y_e, feature_names, dst_path=dst_path, method_type="all"
                    )
                else:
                    logging.info(
                        f"Running only non-deterministic methods for run {run+1}/{runs}."
                    )
                    selection.generate_importances(
                        X_e,
                        y_e,
                        feature_names,
                        dst_path=dst_path,
                        method_type="non_deterministic",
                    )
            else:
                if run == 0 or enable_repeated:
                    logging.info(
                        f"Running all methods for run {run+1}/{runs} and selecting {n} features."
                    )
                    selection.generate_n_importances(
                        X_e,
                        y_e,
                        feature_names,
                        dst_path=dst_path,
                        n=select,
                        method_type="all",
                    )
                else:
                    logging.info(
                        f"Running only non-deterministic methods for run {run+1}/{runs} and selecting {n} features."
                    )
                    selection.generate_n_importances(
                        X_e,
                        y_e,
                        feature_names,
                        dst_path=dst_path,
                        n=select,
                        method_type="non_deterministic",
                    )
