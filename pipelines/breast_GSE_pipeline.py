from pipelines.gse_pipeline import GSEPipeline


def get_pipeline(experiment_id=None, root_dir=""):
    pipeline = GSEPipeline(
        base_dir=f"{root_dir}data/Breast_GSE70947",
        results_dir=f"{root_dir}results/Breast_GSE70947",
        source_data_name="Breast_GSE70947",
        y_col_name="type",
        index_col_name="samples",
        experiment_id=experiment_id,
    )

    return pipeline


if __name__ == "__main__":
    pipeline = get_pipeline()
    pipeline.eval_baseline_df(mode="loo")
    # generate_feature_dfs()
