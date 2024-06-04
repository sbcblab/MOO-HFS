from cec_script import *
import pipelines.arrhythmia_pipeline as arrhythmia_pipeline_

pipelines = [
    arrhythmia_pipeline_.get_pipeline("cec_shuffled"),
]

# Experiment configs
runs = 10
run_ids = [f"run_{i}" for i in range(0, runs)]

# Feature Selection pipeline parameters
min_features = 0
max_features = 50
feature_n_config = [x for x in range(0, max_features + 1)]
cv_k = 5
cv_mode = "kfold"

# NSGAII parameters
cv_k = cv_k
n_max = max_features
n_gen = 10
pop_size = 50
fs_prob = 1.25
evaluator_name = "linearsvm"  # "xgb"
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

    # Generate feature importances using the listed methods
    generate_features(pipeline, runs, run_ids)
    # Evaluate baseline performances of the ranked feature importances using a list of feature counts (feature_n_config)
    eval_features(pipeline, run_ids, feature_n_config, cv_mode, cv_k)
    # Run NSGA-II with the custom operators and the feature importances as input
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
    # Assemble baseline performance results into a consolidated output
    assemble_fs_cv_results(pipeline, run_ids)
    # Assemble MOGA optimization results into a consolidated output
    assemble_nsgaii_results(
        pipeline, run_ids, pop_size, n_gen, cv_k, n_max, fs_prob, evaluator_name
    )
