import pandas as pd
from data_pipelines.gse_pipeline import GSEPipeline


class GencoveReducedPipeline(GSEPipeline):
    def get_source_dfs(self):
        source_df = pd.read_csv(
            self.source_data_file, index_col=self.index_col_name
        ).drop("SKIN-1a6", axis=1)
        y_source_df = (
            source_df[[self.y_col_name]]
            .rename_axis(self.index_col_name)
            .rename(columns={self.y_col_name: "y"})
        )

        return source_df, y_source_df


def get_pipeline(experiment_id=None):
    pipeline = GencoveReducedPipeline(
        base_dir="data/GenesPigmentacaoGencove-20220228-reduced",
        results_dir="results/GenesPigmentacaoGencove-20220228-reduced",
        source_data_name="PELEeOLHO_obs-ABF-24-04",
        y_col_name="EYE-1a6",
        index_col_name="ID",
        experiment_id=experiment_id,
    )

    return pipeline


if __name__ == "__main__":
    pipeline = get_pipeline()
    pipeline.eval_baseline_df(mode="loo")
