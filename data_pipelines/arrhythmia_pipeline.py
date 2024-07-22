import numpy as np
import pandas as pd
from data_pipelines.gse_pipeline import GSEPipeline
import src.feature_generation_tools as fgtools


class ArrhythmiaPipeline(GSEPipeline):
    def get_source_dfs(self):
        source_df = pd.read_csv(self.source_data_file, index_col=self.index_col_name)

        # Replace "?" with actual nulls
        source_df = source_df.replace("?", np.nan).dropna(axis=1)
        y_source_df = source_df[[self.y_col_name]].rename(
            columns={self.y_col_name: "y"}
        )
        source_df = source_df.drop([self.y_col_name], axis=1)

        # Filter out samples of classes containing too few samples (< 5)
        label_counts = y_source_df["y"].value_counts()
        significative_labels = label_counts[label_counts > 5].index
        significative_indexes = y_source_df["y"].isin(significative_labels)
        source_df = source_df.loc[significative_indexes]
        y_source_df = y_source_df.loc[significative_indexes]

        return source_df, y_source_df


def get_pipeline(experiment_id=None, root_dir=""):
    pipeline = ArrhythmiaPipeline(
        base_dir=f"{root_dir}data/Arrhythmia",
        results_dir=f"{root_dir}results/Arrhythmia",
        source_data_name="arrhythmia",
        y_col_name="label",
        index_col_name=None,
        experiment_id=experiment_id,
    )

    return pipeline


if __name__ == "__main__":
    pipeline = get_pipeline()
    pipeline.eval_baseline_df(mode="loo")
