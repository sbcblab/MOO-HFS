from moga_experiment_scripts.base_moga_script import *
from xgboost import XGBClassifier
import data_pipelines.gencove_pipeline as gencove_pipeline_

baseline_fs_pipeline = gencove_pipeline_.get_pipeline("cec_shuffled")

# Experiment configs
runs = 10
run_ids = [f"run_{i}" for i in range(0, runs)]

# Feature Selection pipeline parameters
min_features = 2
max_features = 50
feature_n_config = [x for x in range(0, max_features + 1)]
cv_k = 15
cv_mode = "kfold"

# NSGAII parameters
cv_k = cv_k
n_max = max_features
n_gen = 400
pop_size = 200
fs_prob = 1.25
fitness_evaluator_name = "linearsvm"  # "xgb"
# fs_distrib = {"mrmr": 0.15, "relieff": 0.15, "kruskalwallis": 0.15, "mutualinfo": 0.15, "decisiontree": 0.15, "anovafvalue": 0.15, "randomforest": 0.15}#, "lassocv": 0.15}
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
test_target_metric = "test_f1_macro"

# Execution
experiment = HFSExperiment(
    pipeline=baseline_fs_pipeline,
    runs=runs,
    run_ids=run_ids,
    min_features=min_features,
    max_features=max_features,
    feature_n_config=feature_n_config,
    cv_mode=cv_mode,
    cv_k=cv_k,
    n_gen=n_gen,
    pop_size=pop_size,
    fs_prob=fs_prob,
    fs_distrib=fs_distrib,
    fitness_evaluator_name=fitness_evaluator_name,
)
# Generate feature importances using the listed methods
experiment.generate_features(duplicate_methods)
# Evaluate baseline performances of the ranked feature importances using a list of feature counts (feature_n_config)
experiment.eval_features(test_target_metric, test_clfs)
# Run NSGA-II with the custom operators and the feature importances as input
experiment.run_nsgaii()
# Evaluate NSGA-II results with test classifiers
experiment.eval_nsgaii_features(test_target_metric, test_clfs)
# Assemble baseline performance results into a consolidated output
experiment.assemble_fs_cv_results()
# Assemble MOGA optimization results into a consolidated output (fitness results)
experiment.assemble_nsgaii_results()
# Assemble MOGA evaluation results into a consolidated output
experiment.assemble_nsgaii_evals()