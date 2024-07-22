from experiment_scripts.base_moga_script import *
from xgboost import XGBClassifier
import data_pipelines.arrhythmia_pipeline as arrhythmia_pipeline_

baseline_fs_pipeline = arrhythmia_pipeline_.get_pipeline("cec_shuffled")

# Experiment configs
runs = 10
run_ids = [f"run_{i}" for i in range(0, runs)]

# Feature Selection pipeline parameters
min_features = 2
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
duplicate_methods = [
    "kruskalwallis",
    "anovafvalue",
    "lassocv",
    "relieff",
    "mrmr",
]
test_clfs = {
    "xgb": XGBClassifier()
}  # Define classifiers to evaluate final feature sets, as a dict. Ex.: {"my_clf": MyClassifier()}. If None, the default classifiers are used (LinearSVM).

# Execution
experiment = HFSExperiment(
    baseline_fs_pipeline,
    runs,
    run_ids,
    min_features,
    max_features,
    feature_n_config,
    cv_k,
    cv_mode,
    n_gen,
    pop_size,
    fs_prob,
    evaluator_name,
    fs_distrib,
    duplicate_methods,
    test_clfs,
)
# Generate feature importances using the listed methods
experiment.generate_features()
# Evaluate baseline performances of the ranked feature importances using a list of feature counts (feature_n_config)
experiment.eval_features()
# Run NSGA-II with the custom operators and the feature importances as input
experiment.run_nsgaii()
# Evaluate NSGA-II results with test classifiers
experiment.eval_nsgaii_features()
# Assemble baseline performance results into a consolidated output
experiment.assemble_fs_cv_results()
# Assemble MOGA optimization results into a consolidated output
experiment.assemble_nsgaii_results()
