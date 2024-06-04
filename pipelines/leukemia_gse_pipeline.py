from pipelines.gse_pipeline import GSEPipeline


def get_pipeline(experiment_id=None, root_dir=""):
    pipeline = GSEPipeline(
        base_dir=f"{root_dir}data/Leukemia_GSE28497",
        results_dir=f"{root_dir}results/Leukemia_GSE28497",
        source_data_name="Leukemia_GSE28497",
        y_col_name="type",
        index_col_name="samples",
        experiment_id=experiment_id,
    )

    return pipeline


if __name__ == "__main__":
    pipeline = get_pipeline()
    pipeline.eval_baseline_df(mode="loo")
