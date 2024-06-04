import os
from pathlib import Path
from time import time

import pandas as pd
import src.feature_selection_tools as selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    LeaveOneOut,
    LeavePOut,
    StratifiedKFold,
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from src.commons import *
from src.feature_generation_pipeline import FeatureGenerationPipeline


class GSEPipeline(FeatureGenerationPipeline):
    def __init__(
        self,
        base_dir,
        results_dir,
        source_data_name,
        y_col_name,
        index_col_name,
        experiment_id=None,
    ) -> None:
        self.base_dir = base_dir
        self.results_dir = results_dir
        self.source_data_name = source_data_name
        self.y_col_name = y_col_name
        self.index_col_name = index_col_name

        super().__init__(experiment_id)

    def transform(self):
        pass

    def eval_baseline_df(self, mode="loo", k=5):
        source_df, y_source_df = self.get_source_dfs()

        X_e, y_e, feature_names = self.prepare_dfs(X=source_df, y=y_source_df)

        output_file = f"{self.results_data_dir}/baseline_results/baseline_cv_scores_{mode}-cv_{k}-param.csv"

        cvs = self.evaluate_df(X_e, y_e, output_file=output_file, mode=mode, k=k)

    def get_source_dfs(self):
        logging.info(f"Loading source data file from '{self.source_data_file}'")

        source_df = pd.read_csv(self.source_data_file, index_col=self.index_col_name)
        y_source_df = (
            source_df[[self.y_col_name]]
            .rename_axis(self.index_col_name)
            .rename(columns={self.y_col_name: "y"})
        )

        return source_df, y_source_df

    def get_selected_feature_dfs(self, run_id="run_0"):
        all_dirs = sorted(os.listdir(self.results_dir))
        last_exp = sorted(
            [
                dir
                for dir in all_dirs
                if "cv_scores" not in dir
                and len(os.listdir(f"{self.results_dir}/{dir}")) > 1
            ]
        )[-1]
        df_paths = [
            file
            for file in os.listdir(
                f"{self.results_dir}/{last_exp}/{run_id}/importances"
            )
            if "_importances" in file
        ]
        logging.info(
            f"Loading files from '{self.results_dir}/{last_exp}/{run_id}' directory: {df_paths}"
        )

        dfs = {}
        for df_path in df_paths:
            try:
                dfs[df_path] = pd.read_csv(
                    f"{self.results_dir}/{last_exp}/{run_id}/importances/{df_path}",
                    index_col="feature",
                )
            except Exception as e:
                logging.info(f"Couldn't load file {df_path}. Skipping. Exception: {e}")

        return dfs

    def get_reduced_df(self, source_df, feature_df, force_n=None):
        if force_n > 0:
            logging.info(f"Reducing dataset to top {force_n} features.")
            reduced_df = source_df[list(feature_df[0:force_n].index)]
        else:
            reduced_df = source_df[list(feature_df.index)]

        return reduced_df

    def get_eval_model(self, model="all"):
        clfs = {}
        # clfs["lassocv"] = LassoCV(random_state=42)
        # clfs["decisiontree"] = DecisionTreeClassifier()#random_state=42)
        # clfs["randomforest"] = RandomForestClassifier(n_estimators=150)#, random_state=42)
        clfs["linearsvm"] = LinearSVC(max_iter=3000, dual=True)  # , random_state=42)
        # clfs["logisticregression"] = LogisticRegression(max_iter=3000)#, random_state=42)
        # clfs["knn"] = KNeighborsClassifier(n_neighbors=5)
        # grid = {"n_neighbors": [5,15,25], "weights": ["uniform", "distance"]}

        return clfs

    def evaluate_df(
        self,
        X_e,
        y_e,
        output_file=None,
        filename=None,
        fs_method=None,
        run_id=None,
        mode="loo",
        k=5,
    ):
        clfs = self.get_eval_model()

        if mode == "loo":
            cv = LeaveOneOut()
        elif mode == "lpo":
            cv = LeavePOut(p=k)
        elif mode == "kfold":
            cv = StratifiedKFold(n_splits=k, shuffle=True)

        cv_scores = []
        for clf_name in clfs:
            logging.info(f"Evaluating {clf_name} using {mode} strategy.")
            clf = clfs[clf_name]

            tic_fwd = time()
            scores = cross_validate(
                clf,
                X_e,
                y_e.ravel(),
                cv=cv,
                scoring=[
                    "f1_micro",
                    "f1_macro",
                    "f1_weighted",
                    "precision_micro",
                    "precision_macro",
                    "precision_weighted",
                    "recall_micro",
                    "recall_macro",
                    "recall_weighted",
                    "accuracy",
                ],
                return_estimator=True,
                n_jobs=-1,
            )
            toc_fwd = time()
            logging.info(f"Evaluated {clf_name} in {toc_fwd - tic_fwd:.3f}s")

            logging.debug(f"Scores for {clf_name}: {scores}")
            scores["cv_method"] = mode
            scores["k_param"] = k
            scores["features"] = X_e.shape[1]
            scores["model"] = clf_name
            scores["dataset"] = self.results_dir.split("/")[-1]
            scores["experiment_id"] = self.experiment_id
            scores["run_id"] = run_id
            scores["file"] = filename
            scores["fs_method"] = fs_method
            cv_scores.append(pd.DataFrame.from_dict(scores))

        cv_scores_df = pd.concat(cv_scores)

        logging.debug(f"Final CV scores: {cv_scores}")
        if not output_file:
            output_file = f"{self.results_data_dir}/cv_scores.csv"

        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logging.info(f"Saving scores in: {output_file}")
        cv_scores_df.to_csv(output_file)

        return cv_scores

    def eval_reduced_datasets(
        self,
        run_id,
        source_df,
        y_source_df,
        feature_dfs,
        cv_mode="kfold",
        k=10,
        output_suffix="cv_results",
        feature_n_config=[-1],
    ):
        for feature_df in feature_dfs:
            for n in feature_n_config:
                logging.info(
                    f"Reducing DF with shape {source_df.shape} to lenght {n} out of {len(feature_dfs[feature_df])} using DF {run_id}/{feature_df}"
                )

                try:
                    reduced_df = self.get_reduced_df(
                        source_df, feature_dfs[feature_df], force_n=n
                    )

                    X_e, y_e, feature_names = self.prepare_dfs(
                        X=reduced_df, y=y_source_df, save=False
                    )
                    output_dir = (
                        f"""{self.results_data_dir}/{run_id}/{output_suffix}/"""
                    )
                    filename = feature_df.split(".csv")[0]
                    fs_method = filename.split("_")[0]
                    output_file = f"""{output_dir}/{filename}_scores_{n}-feats_{cv_mode}-cv_{k}-param.csv"""

                    csvs = self.evaluate_df(
                        X_e,
                        y_e,
                        output_file=output_file,
                        fs_method=fs_method,
                        run_id=run_id,
                        mode=cv_mode,
                        k=k,
                    )
                except Exception as e:
                    logging.info(
                        f"Couldn't evaluate {run_id}/{feature_df}. Skipping. \nColumns: {source_df.columns}\nException: {e}"
                    )
