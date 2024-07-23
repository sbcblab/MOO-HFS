from moga_experiment_scripts.base_moga_script import *
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import data_pipelines.arrhythmia_pipeline as arrhythmia_pipeline_

baseline_fs_pipeline = arrhythmia_pipeline_.get_pipeline("sample_run")

# Experiment configs
runs = 1
run_ids = [f"run_{i}" for i in range(0, runs)]

# Feature Selection pipeline parameters
min_features = 2
max_features = 30
feature_n_config = [x for x in range(0, max_features + 1)]
cv_k = 5
cv_mode = "kfold"

# NSGAII parameters
cv_k = cv_k  # Number of folds for cross-validation
n_max = max_features  # Maximum number of features to consider
n_gen = 50  # Number of generations
pop_size = 50  # Population size
fs_prob = 1.25  # Probability multiplier for the selection of a feature based on importance. Higher values decrease the influence of the importance value in the selection of a feature.
fitness_evaluator_name = "xgb"  # Internal evaluator name to record during the optimization process.
fitness_evaluator_obj = XGBClassifier()  # Internal evaluator to use for the optimization process. Accepts the SKLearn predictor format.
fitness_target_metric = "test_f1_macro"

# Define the distribution of feature selection methods to be used in the optimization process. The sum of the values must be 1.
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

# Inform methods to replicate runs, if necessary (stochastic methods)
duplicate_methods = [
    "kruskalwallis",
    "anovafvalue",
    "lassocv",
    "relieff",
    "mrmr",
]

# Define classifiers to evaluate final feature sets, as a dict. Ex.: {"my_clf": MyClassifier()}. If None, the default classifiers are used (LinearSVM).
test_clfs = {"xgb": XGBClassifier()}
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
    fitness_evaluator_obj=fitness_evaluator_obj,
    fitness_target_metric=fitness_target_metric,
)
# Generate feature importances using baseline methods
experiment.generate_baseline_features(duplicate_methods)
# Evaluate baseline performances of the ranked feature importances using a list of feature counts (feature_n_config)
experiment.eval_baseline_features(test_target_metric, test_clfs)
# Run NSGA-II with the custom operators and the feature importances as input
experiment.run_moga_optimization()
# Evaluate NSGA-II results with test classifiers
experiment.eval_moga_features(test_target_metric, test_clfs)
# Assemble baseline performance results into a consolidated output
experiment.assemble_baseline_cv_results()
# Assemble MOGA optimization results into a consolidated output (fitness results)
experiment.assemble_moga_fitness_results()
# Assemble MOGA evaluation results into a consolidated output
experiment.assemble_moga_evals()
