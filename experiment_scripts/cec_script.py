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


def generate_features(pipeline, runs, run_ids):
    X, y = pipeline.get_source_dfs()

    # Generate feature importances
    select = -1
    pipeline.generate_feature_dfs(X, y, runs, select, enable_repeated=False)

    ## The following section is a workaround to not recalculating deterministic methods (which always yield the same results)
    deterministic_methods = [
        "kruskalwallis",
        "anovafvalue",
        "lassocv",
        "relieff",
        "mrmr",
    ]

    csvs_to_replicate = [
        csv
        for csv in os.listdir(f"{pipeline.results_data_dir}/run_0/importances")
        if any([x in csv for x in deterministic_methods])
    ]
    for run_id in run_ids:
        if run_id != "run_0":
            for csv in csvs_to_replicate:
                src = f"{pipeline.results_data_dir}/run_0/importances/{csv}"
                dst = f"{pipeline.results_data_dir}/{run_id}/importances/{csv}"
                shutil.copy(src, dst)


def eval_features(pipeline, run_ids, feature_n_config, cv_mode, cv_k):
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
    evaluator_name="linearsvm",
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
):
    # Load feature importances
    output_suffix = f"""nsgaii_{pipeline.source_data_name}_{n_gen}-gens_{pop_size}-pop_{cv_k}-k_{n_max}-feat_{str(prob_multiplier).replace(".", "")}-prob_{evaluator_name}"""
    output_path = (
        f"""{pipeline.results_data_dir}/{run_id}/nsgaii_solutions/{output_suffix}/"""
    )

    logging.info(f"Run results will be saved in {output_path}.")

    logging.info(f"Loading feature importance sets for {run_id}")
    feature_dfs = pipeline.get_selected_feature_dfs(run_id=run_id)
    logging.info([filename for filename in feature_dfs])
    feature_dfs = {
        filename.split("_")[0]: feature_dfs[filename] for filename in feature_dfs
    }
    methods = [method for method in fs_distrib]

    logging.info(f"Creating problem for {run_id}")
    # Update reference feature_dfs
    ma_problem = MicroarrayProblem(df, y, cv_k, n_max)
    ma_problem.feature_dfs = feature_dfs
    ma_problem.fs_distrib = fs_distrib

    # Define the algorithm
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.mutation.bitflip import BFM
    from pymoo.optimize import minimize

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FeatureSampling(
            max_attempts=100, prob_multiplier=prob_multiplier, min_features=2
        ),  # MySampling(),
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
        # seed=1,
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
    out_df.to_csv(f"{output_path}/full_pop_{output_suffix}.csv")

    # Save full population features
    out_df = pd.DataFrame([indiv.X for indiv in res.pop], columns=feature_names)
    out_df.to_csv(f"""{output_path}/feature_sets_{output_suffix}.csv""")

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
    out_df.to_csv(f"{output_path}/front_{output_suffix}.csv")

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
    out_df.to_csv(f"{output_path}/full_pop_hist_{output_suffix}.csv")

    # Save hypervolume
    hyperv_array = np.array([hv, n_evals]).T

    out_df = pd.DataFrame(hyperv_array, columns=["hypervolume", "n_evals"])
    out_df.to_csv(f"{output_path}/hypervolume_{output_suffix}.csv")

    return True


def assemble_nsgaii_results(
    pipeline, run_ids, pop_size, n_gen, cv_k, n_max, prob_multiplier, evaluator_name
):
    # Recover all files from NSGAII
    # TODO: remove this expression
    # subdir_iterators = [os.walk(f"{pipeline.results_data_dir}/{run_id}/nsgaii_solutions") for pipeline in pipelines for run_id in run_ids]
    subdir_iterators = [
        os.walk(f"{pipeline.results_data_dir}/{run_id}/nsgaii_solutions")
        for run_id in run_ids
    ]
    subdir_files = [
        os.path.join(dirpath, filename)
        for subdir_iterator in subdir_iterators
        for (dirpath, dirnames, filenames) in subdir_iterator
        for filename in filenames
    ]
    expected_output_suffix = f"""front_nsgaii_{pipeline.source_data_name}_{n_gen}-gens_{pop_size}-pop_{cv_k}-k_{n_max}-feat_{str(prob_multiplier).replace(".", "")}-prob_{evaluator_name}"""
    logging.info(f"Looking for files with suffix: {expected_output_suffix}.")

    datasets = [
        subdir_file
        for subdir_file in subdir_files
        if (expected_output_suffix in subdir_file)
    ]
    logging.info(f"Recovering all front files: {datasets}.")

    dfs = []
    for ds in datasets:
        df = pd.read_csv(ds)
        dfs.append(df)

    dfs = pd.concat(dfs)

    dfs["test_f1_macro"] = 1 - dfs["test_f1_macro"]

    # Recover all files from pipelines
    # other_cv_dfs = pd.concat([pd.read_csv(f"{pipeline.results_data_dir}/baseline_results/cv_results_assembled.csv") for pipeline in [pipeline]])
    other_cv_dfs = pd.read_csv(
        f"{pipeline.results_data_dir}/baseline_results/cv_results_assembled.csv"
    )

    # Assemble all files into one huge report
    # logging.info("Saving final assembled results in ", f"./results/front_assembled_extended_{n_gen}gens_{cv_k}k_{n_max}feat.csv")
    # pd.concat([other_cv_dfs, dfs]).to_csv(f"./results/front_assembled_extended_{n_gen}gens_{cv_k}k_{n_max}feat.csv")
    output_suffix = f"""nsgaii_{pipeline.source_data_name}_{n_gen}-gens_{pop_size}-pop_{cv_k}-k_{n_max}-feat_{str(prob_multiplier).replace(".", "")}-prob_{evaluator_name}_{len(run_ids)}-runs"""

    logging.info(
        f"Saving final assembled results in {pipeline.results_data_dir}/front_assembled_extended_{output_suffix}.csv"
    )
    pd.concat([other_cv_dfs, dfs]).to_csv(
        f"{pipeline.results_data_dir}/front_assembled_extended_{output_suffix}.csv"
    )
