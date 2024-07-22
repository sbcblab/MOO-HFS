import shutil
import pandas as pd
import numpy as np
import os

from src.nsgaii import (
    BinaryCrossover,
    FeatureSampling,
    MicroarrayProblem,
    FeatureSamplingMutation,
)
from src.commons import *
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)


class HFSExperiment:
    def __init__(
        self,
        pipeline,
        runs,
        run_ids,
        min_features,
        max_features,
        feature_n_config,
        cv_mode,
        cv_k,
        n_gen,
        pop_size,
        fs_prob,
        fs_distrib,
        fitness_evaluator_name,
    ):
        self.pipeline = pipeline
        self.runs = runs
        self.run_ids = run_ids
        self.min_features = min_features
        self.max_features = max_features
        self.feature_n_config = feature_n_config
        self.cv_mode = cv_mode
        self.cv_k = cv_k
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.fs_prob = fs_prob
        self.fs_distrib = fs_distrib
        self.fitness_evaluator_name = fitness_evaluator_name

        self.get_experiment_name()

    def get_experiment_name(self):
        self.experiment_name = f"""nsgaii_{self.pipeline.source_data_name}_{self.n_gen}-gens_{self.pop_size}-pop_{self.cv_k}-k_{self.min_features}-to-{self.max_features}-feat_{str(self.fs_prob).replace(".", "")}-prob_{self.fitness_evaluator_name}"""

        return self.experiment_name

    def generate_features(self, duplicate_methods=None):
        generate_features(self.pipeline, self.runs, self.run_ids, duplicate_methods=duplicate_methods)

    def eval_features(self, target_metric, clfs):
        eval_features(
            self.pipeline,
            self.run_ids,
            self.feature_n_config,
            self.cv_mode,
            self.cv_k,
            clfs,
        )

    def run_nsgaii(self):
        run_nsgaii(
            self.pipeline,
            self.run_ids,
            self.pop_size,
            self.n_gen,
            self.fs_distrib,
            self.max_features,
            self.cv_k,
            self.fs_prob,
            self.fitness_evaluator_name,
            self.experiment_name,
        )

    def eval_nsgaii_features(self, target_metric, clfs):
        front_paths, feature_set_paths = get_file_paths(
            self.pipeline, self.experiment_name, self.run_ids
        )

        eval_nsgaii_features(
            self.pipeline,
            target_metric,
            self.cv_k,
            clfs=clfs,
            front_paths=front_paths,
            feature_set_paths=feature_set_paths,
        )

    def assemble_fs_cv_results(self):
        assemble_fs_cv_results(self.pipeline, self.run_ids)

    def assemble_nsgaii_results(self):
        front_paths, feature_set_paths = get_file_paths(
            self.pipeline, self.experiment_name, self.run_ids
        )
                
        assemble_nsgaii_results(self.pipeline, front_paths, self.experiment_name)

    def assemble_nsgaii_evals(self):
        assemble_nsgaii_evals(self.pipeline, self.experiment_name)

def generate_features(pipeline, runs, run_ids, duplicate_methods=None):
    X, y = pipeline.get_source_dfs()

    # Generate feature importances
    select = -1
    pipeline.generate_feature_dfs(X, y, runs, select, enable_repeated=False)

    ## The following section is a workaround to not recalculating deterministic methods (which always yield the same results)
    if duplicate_methods is not None:
        logging.info(f"Duplicating runs for methods {duplicate_methods}")
        csvs_to_replicate = [
            csv
            for csv in os.listdir(f"{pipeline.results_data_dir}/run_0/importances")
            if any([x in csv for x in duplicate_methods])
        ]
        for run_id in run_ids:
            if run_id != "run_0":
                for csv in csvs_to_replicate:
                    src = f"{pipeline.results_data_dir}/run_0/importances/{csv}"
                    dst = f"{pipeline.results_data_dir}/{run_id}/importances/{csv}"
                    shutil.copy(src, dst)


def eval_features(pipeline, run_ids, feature_n_config, cv_mode, cv_k, clfs=None):
    # Evaluate datasets
    X, y = pipeline.get_source_dfs()

    for run_id in run_ids:
        # Retrieve feature dfs
        feature_dfs = pipeline.get_selected_feature_dfs(run_id=run_id)

        # Run evaluation
        pipeline.eval_reduced_datasets(
            run_id=run_id,
            source_df=X,
            y_source_df=y,
            feature_dfs=feature_dfs,
            cv_mode=cv_mode,
            k=cv_k,
            feature_n_config=feature_n_config,
            clfs=clfs,
        )


def assemble_fs_cv_results(pipeline, run_ids):
    # Assemble all results into one dataset
    experiment_dfs = []
    for run_id in run_ids:
        output_dir = f"""{pipeline.results_data_dir}/{run_id}/cv_results/"""
        csv_names = [x for x in os.listdir(output_dir) if "_scores_" in x]

        run_dfs = [pd.read_csv(f"{output_dir}/{csv_name}") for csv_name in csv_names]
        run_dfs = pd.concat(run_dfs)
        run_dfs["run_id"] = run_id
        run_dfs["experiment_id"] = pipeline.experiment_id
        experiment_dfs.append(run_dfs)

    experiment_dfs = pd.concat(experiment_dfs)

    if not os.path.exists(f"""{pipeline.results_data_dir}/baseline_results/"""):
        os.makedirs(f"""{pipeline.results_data_dir}/baseline_results/""")
    experiment_dfs.to_csv(
        f"""{pipeline.results_data_dir}/baseline_results/cv_results_assembled.csv"""
    )


def run_nsgaii(
    pipeline,
    run_ids,
    pop_size,
    n_gen,
    fs_distrib,
    n_max,
    cv_k,
    prob_multiplier,
    evaluator_name,
    experiment_name,
):
    ###############################################################

    # Improve solutions with NSGAII

    # Create the problem definition to be solved

    X, y = pipeline.get_source_dfs()

    X_e, y_e, feature_names = pipeline.prepare_dfs(X=X, y=y, save=True)
    df = pd.DataFrame(X_e, columns=feature_names, index=X.index)
    logging.info(f"Running NSGAII for dataset with shape {df.shape} for runs {run_ids}")

    for run_id in run_ids:
        run_nsgaii_iter(
            pipeline,
            df,
            y,
            feature_names,
            run_id,
            pop_size,
            n_gen,
            fs_distrib,
            n_max,
            cv_k,
            prob_multiplier,
            evaluator_name,
            experiment_name,
        )


def run_nsgaii_iter(
    pipeline,
    df,
    y,
    feature_names,
    run_id,
    pop_size,
    n_gen,
    fs_distrib,
    n_max,
    cv_k,
    prob_multiplier,
    evaluator_name,
    experiment_name,
):
    # Load feature importances
    output_path = (
        f"""{pipeline.results_data_dir}/{run_id}/nsgaii_solutions/{experiment_name}/"""
    )

    logging.info(f"Run results will be saved in {output_path}.")

    logging.info(f"Loading feature importance sets for {run_id}")
    feature_dfs = pipeline.get_selected_feature_dfs(run_id=run_id)
    logging.info(f"Files loaded: {[filename for filename in feature_dfs]}")
    feature_dfs = {
        filename.split("_")[0]: feature_dfs[filename] for filename in feature_dfs
    }

    logging.info(f"Creating problem for {run_id}")
    # Update reference feature_dfs
    ma_problem = MicroarrayProblem(df, y, cv_k, n_max)
    ma_problem.feature_dfs = feature_dfs
    ma_problem.fs_distrib = fs_distrib

    # Define the algorithm
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FeatureSampling(
            max_attempts=100, prob_multiplier=prob_multiplier, min_features=2
        ),
        crossover=BinaryCrossover(),
        mutation=FeatureSamplingMutation(),
        eliminate_duplicates=True,
    )

    logging.info(f"Minimizing objectives for {run_id}")
    # Optimize the solutions
    res = minimize(
        ma_problem,
        algorithm,
        ("n_gen", n_gen),
        verbose=True,
        save_history=True,
        evaluator_name=evaluator_name,
    )

    logging.info(f"Storing results for {run_id}")
    # Prepare historical data
    hist = res.history

    n_evals = []  # corresponding number of function evaluations\
    hist_F = []  # the objective space values in each generation
    hist_cv = []  # constraint violation in each generation
    hist_cv_avg = []  # average constraint violation in the whole population

    for algo in hist:
        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algo.opt

        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())

        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])

    # Calculate the hypervolume
    from pymoo.indicators.hv import Hypervolume
    import matplotlib.pyplot as plt

    approx_ideal = np.min([[indiv.F[0], indiv.F[1]] for indiv in res.pop], axis=0)
    approx_nadir = np.max([[indiv.F[0], indiv.F[1]] for indiv in res.pop], axis=0)

    metric = Hypervolume(
        ref_point=np.array([1.05, 1.05]),
        norm_ref_point=False,
        zero_to_one=True,
        ideal=approx_ideal,
        nadir=approx_nadir,
    )

    hv = [metric.do(_F) for _F in hist_F]

    F = res.F
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

    fl = nF.min(axis=0)
    fu = nF.max(axis=0)

    # Prepare to save outputs
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save full population
    full_pop_result = [[indiv.F[0], indiv.F[1]] for indiv in res.pop]

    out_df = pd.DataFrame(full_pop_result, columns=["test_f1_macro", "features"])
    out_df["cv_method"] = "kfold"
    out_df["k_param"] = cv_k
    out_df["model"] = evaluator_name
    out_df["dataset"] = pipeline.results_dir.split("/")[-1]
    out_df["experiment_id"] = pipeline.experiment_id
    out_df["run_id"] = run_id
    out_df["file"] = ""
    out_df["fs_method"] = f"NSGA-II_{n_gen}gen"
    out_df["fit_time"] = res.exec_time
    out_df.to_csv(f"{output_path}/full_pop.csv")

    # Save full population features
    out_df = pd.DataFrame([indiv.X for indiv in res.pop], columns=feature_names)
    out_df.to_csv(f"""{output_path}/feature_sets.csv""")

    # Save front
    out_df = pd.DataFrame(F, columns=["test_f1_macro", "features"])
    out_df["cv_method"] = "kfold"
    out_df["k_param"] = cv_k
    out_df["model"] = evaluator_name
    out_df["dataset"] = pipeline.results_dir.split("/")[-1]
    out_df["experiment_id"] = pipeline.experiment_id
    out_df["run_id"] = run_id
    out_df["file"] = ""
    out_df["fs_method"] = f"NSGA-II_{n_gen}gen"
    out_df["fit_time"] = res.exec_time
    out_df.to_csv(f"{output_path}/front.csv")

    # Save all population scores in all generations
    full_pop_hist = [
        [n, i, indiv.F[0], indiv.F[1]]
        for n, algo in enumerate(hist)
        for i, indiv in enumerate(algo.pop)
    ]

    out_df = pd.DataFrame(
        full_pop_hist, columns=["gen", "gen_index", "test_f1_macro", "features"]
    )
    out_df["cv_method"] = "kfold"
    out_df["k_param"] = cv_k
    out_df["model"] = evaluator_name
    out_df["dataset"] = pipeline.results_dir.split("/")[-1]
    out_df["experiment_id"] = pipeline.experiment_id
    out_df["run_id"] = run_id
    out_df["file"] = ""
    out_df["fs_method"] = f"NSGA-II_{n_gen}gen"
    out_df["fit_time"] = res.exec_time
    out_df.to_csv(f"{output_path}/full_pop_hist.csv")

    # Save hypervolume
    hyperv_array = np.array([hv, n_evals]).T

    out_df = pd.DataFrame(hyperv_array, columns=["hypervolume", "n_evals"])
    out_df.to_csv(f"{output_path}/hypervolume.csv")

    return True


def eval_fronts_feature_sets(front_df, feature_list_df, X, y, clf=None, cv_k=5):
    evaluations = []

    for front_id in front_df.index:
        feature_list = (
            feature_list_df.loc[front_id]
            .where(feature_list_df.loc[front_id] == True)
            .dropna()
            .index
        )

        X_selected = X[feature_list].copy()
        y = y

        scores = evaluate(X_selected, y, clf, cv_k)
        evaluations.append({"front_id": front_id, "scores": scores})

    return evaluations

def get_target_from_front_evals(evals, target):
    metric = [(item["front_id"], np.mean(item["scores"][target])) for item in evals]

    return metric


def evaluate(X_e, y_e, clf, cv_k):
    cv = StratifiedKFold(n_splits=cv_k, shuffle=True)

    scores = cross_validate(
        clf,
        X_e,
        y_e,
        cv=cv,
        scoring=[
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "recall_micro",
            "recall_macro",
            "recall_weighted",
            "accuracy",
        ],
        return_estimator=False,
        n_jobs=-1,
    )

    return scores


def eval_nsgaii_features(
    pipeline, target_metric, cv_k, clfs, front_paths, feature_set_paths
):
    X, y = pipeline.get_source_dfs()

    front_dfs = [pd.read_csv(front_path, index_col=0) for front_path in front_paths]
    feature_set_dfs = [
        pd.read_csv(feature_set_path, index_col=0)
        for feature_set_path in feature_set_paths
    ]
    new_front_dfs = []

    X_e, y_e, feature_names = pipeline.prepare_dfs(X=X, y=y, save=False)
    df = pd.DataFrame(X_e, columns=feature_names, index=X.index)

    for clf_name, clf in clfs.items():
        for front_df, feature_list_df in zip(front_dfs, feature_set_dfs):
            evals = eval_fronts_feature_sets(
                front_df, feature_list_df, df, y_e, clf, cv_k
            )
            scores = get_target_from_front_evals(evals, target_metric)

            target_metrics_df = pd.DataFrame(
                scores, columns=["front_id", target_metric]
            )
            target_metrics_df["model"] = clf_name

            front_df[target_metric] = target_metrics_df[target_metric]
            front_df["model"] = clf_name
            new_front_dfs.append(front_df)

    front_evals_df = pd.concat(new_front_dfs)

    # Store results
    if not os.path.exists(f"""{pipeline.results_data_dir}/nsgaii_results/"""):
        os.makedirs(f"""{pipeline.results_data_dir}/nsgaii_results/""")
    front_evals_df.to_csv(
        f"""{pipeline.results_data_dir}/nsgaii_results/cv_results_assembled.csv"""
    )


def get_file_paths(pipeline, experiment_name, run_ids):
    for run_id in run_ids:
        subdir_iterators = [
            os.walk(
                f"{pipeline.results_data_dir}/{run_id}/nsgaii_solutions/{experiment_name}"
            )
            for run_id in run_ids
        ]
        subdir_files = [
            os.path.join(dirpath, filename)
            for subdir_iterator in subdir_iterators
            for (dirpath, dirnames, filenames) in subdir_iterator
            for filename in filenames
        ]

        expected_dir_name = "front"
        logging.info(f"Looking for files with prefix: {expected_dir_name}.")

        front_paths = [
            subdir_file
            for subdir_file in subdir_files
            if (expected_dir_name in subdir_file)
        ]
        logging.info(f"Recovering all front files: {front_paths}.")

        expected_dir_name = "feature_sets"
        logging.info(f"Looking for files with suffix: {expected_dir_name}.")

        feature_set_paths = [
            subdir_file
            for subdir_file in subdir_files
            if (expected_dir_name in subdir_file)
        ]
        logging.info(f"Recovering all front files: {feature_set_paths}.")

        return front_paths, feature_set_paths


def assemble_nsgaii_results(pipeline, front_paths, experiment_name):

    dfs = []
    for ds in front_paths:
        df = pd.read_csv(ds)
        dfs.append(df)

    dfs = pd.concat(dfs)

    dfs["test_f1_macro"] = 1 - dfs["test_f1_macro"]

    # Recover all files from pipelines
    other_cv_dfs = pd.read_csv(
        f"{pipeline.results_data_dir}/baseline_results/cv_results_assembled.csv"
    )

    # Assemble all files into one huge report
    logging.info(
        f"Saving final assembled results in {pipeline.results_data_dir}/front_assembled_extended_{experiment_name}.csv"
    )
    pd.concat([other_cv_dfs, dfs]).to_csv(
        f"{pipeline.results_data_dir}/front_assembled_extended_{experiment_name}.csv"
    )


def assemble_nsgaii_evals(pipeline, experiment_name):

    # Recover nsgaii eval results
    moga_cv_dfs = pd.read_csv(
        f"{pipeline.results_data_dir}/nsgaii_results/cv_results_assembled.csv"
    )

    # Recover baseline results
    baseline_cv_dfs = pd.read_csv(
        f"{pipeline.results_data_dir}/baseline_results/cv_results_assembled.csv"
    )

    # Assemble all files into one huge report
    report_path = f"{pipeline.results_data_dir}/complete_evals_{experiment_name}.csv"
    logging.info(
        f"Saving final assembled results in {report_path}"
    )
    pd.concat([baseline_cv_dfs, moga_cv_dfs]).to_csv(
        report_path
    )
