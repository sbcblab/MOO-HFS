from cec_script import *
import pipelines.breast_gse_pipeline as breast_gse_pipeline_

pipelines = [
    breast_gse_pipeline_.get_pipeline("cec_shuffled"),
]

# Experiment configs
runs = 10
run_ids = [f"run_{i}" for i in range(0, runs)]

# Feature Selection pipeline parameters
max_features = 50
feature_n_config = [x for x in range(0, max_features + 1)]
cv_k = 15
cv_mode = "kfold"

# NSGAII parameters
cv_k = cv_k
n_max = max_features
n_gen = 200
pop_size = 200
fs_prob = 1.25
evaluator_name = "linearsvm"  # "xgb"
# fs_distrib = {"mrmr": 0.15, "relieff": 0.15, "kruskalwallis": 0.15, "mutualinfo": 0.15, "decisiontree": 0.15, "anovafvalue": 0.15, "randomforest": 0.15} #, "lassocv": 0.15}
fs_distrib = {
    "mrmr": 0.125,
    "relieff": 0.125,
    "kruskalwallis": 0.125,
    "mutualinfo": 0.125,
    "decisiontree": 0.125,
    "anovafvalue": 0.125,
    "randomforest": 0.125,
    "lassocv": 0.125,
}

# Execution
for pipeline in pipelines:
    generate_features(pipeline, runs, run_ids)
    eval_features(pipeline, run_ids, feature_n_config, cv_mode, cv_k)
    run_nsgaii(
        pipeline,
        run_ids,
        pop_size,
        n_gen,
        fs_distrib,
        n_max,
        cv_k,
        fs_prob,
        evaluator_name,
    )
    assemble_fs_cv_results(pipeline, run_ids)
    assemble_nsgaii_results(
        pipeline, run_ids, pop_size, n_gen, cv_k, n_max, fs_prob, evaluator_name
    )
