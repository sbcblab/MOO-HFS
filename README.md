# MOO-HFS
 A framework for Multi-Objective Optimization of Hybrid Feature Selection in tabular data.

tldr: uses baseline feature selection methods to provide information for a multi-objective genetic algorithm to optimize. The GA combines and modifies the feature sets in order to imrpove classification metrics.


## How to use

### Example:
 - Check the script in ´moga_experiment_scripts\sample_run_arrhyth.py´ for a quick example. Run the script and check parameters.

### Guidelines:
Two steps are necessary to start:
 1) Define a **data_pipeline** using the base **GSEPipeline** class to load and prepare data for prediction. Example: data_pipelines\arrhythmia_pipeline.py
 2) Define a script to call a **HFSExperiment** instance, define your parameters and execute the desired steps. Example: ´moga_experiment_scripts\eval_script_arrhyth.py´

### General HFSExperiment class functions:
- generate_baseline_features: Generate feature importances using baseline methods.
- evaluate_baseline_features: Evaluate baseline performances of the ranked feature importances using a list of feature counts (feature_n_config).
- run_moga_optimization: Run NSGA-II with the custom operators and the feature importances as input.
- eval_moga_features: Evaluate NSGA-II results with test classifiers.
- assemble_baseline_cv_results: Assemble baseline performance results into a consolidated output.
- assemble_moga_fitness_results: Assemble MOGA optimization results into a consolidated output (fitness results).
- assemble_moga_evals: Assemble MOGA evaluation results by a test classifier into a consolidated output.

### General tips:
- Avoid columns with null values or fill them using an adequate approach, otherwise they **will** be discarded during the process.
- Consider using a decent balance between numbers of generations and individuals.
- Consider having at least a couple individuals per difference of min_features and max_features.
- Consider using different fitness evaluators and test evaluators. Certain classifiers work best for certain data types.