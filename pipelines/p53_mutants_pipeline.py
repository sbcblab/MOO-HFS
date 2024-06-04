import pandas as pd
from pipelines.gse_pipeline import GSEPipeline
import src.feature_generation_tools as fgtools


class p53MutantsPipeline(GSEPipeline):
    def get_source_dfs(self):
        source_df = pd.read_csv(self.source_data_file, index_col=self.index_col_name)
        source_df = source_df.drop(
            source_df[source_df["feature_0"] == "?"].index, axis=0
        )
        source_df = source_df[~source_df[self.y_col_name].isna()]

        y_source_df = (
            source_df[[self.y_col_name]]
            .rename_axis(self.index_col_name)
            .rename(columns={self.y_col_name: "y"})
        )

        source_df = source_df.drop(["K9_label", "K8_label"], axis=1).astype("float")

        return source_df, y_source_df


def get_pipeline(experiment_id=None, root_dir=""):
    pipeline = p53MutantsPipeline(
        base_dir=f"{root_dir}data/p53_Mutants",
        results_dir=f"{root_dir}results/p53_Mutants",
        source_data_name="p53_Mutants",
        y_col_name="K8_label",
        index_col_name=0,
        experiment_id=experiment_id,
    )

    return pipeline


if __name__ == "__main__":
    pipeline = get_pipeline()
    pipeline.eval_baseline_df(mode="loo")
