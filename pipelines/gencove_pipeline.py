import pandas as pd
from pipelines.gse_pipeline import GSEPipeline
import src.feature_generation_tools as fgtools


class GencoveReducedPipeline(GSEPipeline):
    def get_source_dfs(self):
        source_df = pd.read_csv(self.source_data_file, index_col=self.index_col_name)
        y_source_df = (
            source_df[[self.y_col_name]]
            .rename_axis(self.index_col_name)
            .rename(columns={self.y_col_name: "y"})
        )

        source_df = fgtools.split_features(source_df, self.y_col_name)

        return source_df, y_source_df


def get_pipeline(experiment_id=None, root_dir=""):
    pipeline = GencoveReducedPipeline(
        base_dir=f"{root_dir}data/GenesPigmentacaoGencove-20220228",
        results_dir=f"{root_dir}results/GenesPigmentacaoGencove-20220228",
        source_data_name="Genes-Pigmentacao-63009-Eye-SemMissing-3-Classes",
        y_col_name="3-Classes",
        index_col_name="id",
        experiment_id=experiment_id,
    )

    return pipeline


if __name__ == "__main__":
    pipeline = get_pipeline()
    pipeline.eval_baseline_df(mode="loo")
