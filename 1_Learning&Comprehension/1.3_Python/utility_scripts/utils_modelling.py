"""
Module that contains utility functions for Modelling:
    1. classification_model_evaluation_track: Function to track performance metrics
    for classification experiments.
    2. regression_model_evaluation_track: Function to track performance metrics
    for regression experiments.
    3. DropCorrelatedFeatures: Class to inspect high correlation feature pairs and
    drop according to a strategy. Has seperate functions for both numerical and
    categorical features.
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.inspection import permutation_importance
from sklearn.metrics import RocCurveDisplay, auc

def classification_model_evaluation_track(
    eval_metric2_val: dict = None,
    model_name: str = None,
    feature_count: int = None,
    accuracy_score: tuple[float, float] = (None, None),
    precision_score: tuple[float, float] = (None, None),
    recall_score: tuple[float, float] = (None, None),
    bal_accuracy_score: tuple[float, float] = (None, None),
    f1_score: tuple[float, float] = (None, None),
    roc_auc: tuple[float, float] = (None, None),
    fit_time: float = None,
    sort_by: str = "test_accuracy",
) -> pd.DataFrame:
    """
    Function to track performance metrics for classification experiments.

    Args:
        eval_metric2_val : dict, optional
            Dictionary populated with the necessary classification metric scores.
            No need to pass if it is the first experiment. The default is None.
        model_name : str, optional
            The model name to be used to keep track of the experiment.
            The default is None.
        feature_count : int, optional
            The number of features used to fit the model.
            The default is None.
        accuracy_score : tuple[float, float], optional
            Accuracy score of the experiment in the order (train_score, test_score).
            The default is (None, None).
        recall_score : tuple[float, float], optional
            Recall score of the experiment in the order (train_score, test_score).
            The default is (None, None).
        bal_accuracy_score : tuple[float, float], optional
            Balanced accuracy score of the experiment in the order (train_score, test_score).
            The default is (None, None).
        f1_score : tuple[float, float], optional
            F1 score of the experiment in the order (train_score, test_score).
            The default is (None, None).
        roc_auc : tuple[float, float], optional
            Roc area under curve of the experiment in the order (train_score, test_score).
            The default is (None, None).
        fit_time : float, optional
            Model fit time of the experiment in the order (train_score, test_score).
            The default is None.
        sort_by : str, optional
            The metric to be used to sort the output dataframe.
            The default is 'test_accuracy'.

    Returns:
        pd.DataFrame
            The dataframe populated with the classification metrics mentioned
            as parameters and sorted by the metric listed in the sort_by parameter.

    Examples:
        >>> df_test = classification_model_evaluation_track(model_name='test1', feature_count=9, recall_score=(.99, .85), roc_auc=(0.56721, 0.55116), fit_time=11.2)
        >>> print(df_test) # doctest: +ELLIPSIS
          model_name  feature_count  ... test_roc_auc train_fit_time
        0      test1              9  ...       0.5512           11.2
        <BLANKLINE>
        [1 rows x 15 columns]

        >>> df_test = classification_model_evaluation_track(eval_metric2_val=df_test.to_dict('list'), model_name='test2', feature_count=12, accuracy_score=(.911, .886), recall_score=(0.2341, 0.3121), fit_time=5.2)
        >>> print(df_test) # doctest: +ELLIPSIS
          model_name  feature_count  ...  test_roc_auc  train_fit_time
        1      test2             12  ...           NaN             5.2
        0      test1              9  ...        0.5512            11.2
        <BLANKLINE>
        [2 rows x 15 columns]
    """

    if eval_metric2_val is None:
        eval_metric2_val = {
            "model_name": [],
            "feature_count": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "train_precision": [],
            "test_precision": [],
            "train_recall": [],
            "test_recall": [],
            "train_bal_accuracy": [],
            "test_bal_accuracy": [],
            "train_f1_score": [],
            "test_f1_score": [],
            "train_roc_auc": [],
            "test_roc_auc": [],
            "train_fit_time": [],
        }

    eval_metric2_val["model_name"].append(model_name)
    eval_metric2_val["feature_count"].append(feature_count)
    eval_metric2_val["train_accuracy"].append(accuracy_score[0])
    eval_metric2_val["test_accuracy"].append(accuracy_score[1])
    eval_metric2_val["train_precision"].append(precision_score[0])
    eval_metric2_val["test_precision"].append(precision_score[1])
    eval_metric2_val["train_recall"].append(recall_score[0])
    eval_metric2_val["test_recall"].append(recall_score[1])
    eval_metric2_val["train_bal_accuracy"].append(bal_accuracy_score[0])
    eval_metric2_val["test_bal_accuracy"].append(bal_accuracy_score[1])
    eval_metric2_val["train_f1_score"].append(f1_score[0])
    eval_metric2_val["test_f1_score"].append(f1_score[1])
    eval_metric2_val["train_roc_auc"].append(roc_auc[0])
    eval_metric2_val["test_roc_auc"].append(roc_auc[1])
    eval_metric2_val["train_fit_time"].append(fit_time)

    evaluation_df = pd.DataFrame(eval_metric2_val)

    return evaluation_df.sort_values(by=sort_by, ascending=False).round(4)


######################################


def regression_model_evaluation_track(
    eval_metric2_val: dict = None,
    model_name: str = None,
    feature_count: int = None,
    explained_variance: tuple[float, float] = (None, None),
    neg_mean_absolute_error: tuple[float, float] = (None, None),
    neg_mean_squared_error: tuple[float, float] = (None, None),
    r2: tuple[float, float] = (None, None),
    fit_time: float = None,
    sort_by: str = "train_neg_mean_squared_error",
) -> pd.DataFrame:
    """
    Function to track performance metrics for regression experiments.
    To understand "neg_xxx_xxx", refer:
        https://stackoverflow.com/questions/48244219/is-sklearn-metrics-mean-squared-error-the-larger-the-better-negated

    Args:
        eval_metric2_val : dict, optional
            Dictionary populated with the necessary classification metric scores.
            The default is None.
        model_name : str, optional
            The model name to be used to keep track of the experiment.
            The default is None.
        feature_count : int, optional
            The number of features used to fit the model.
            The default is None.
        explained_variance : tuple[float, float], optional
            Explained variance regression score function in the order (train_score, test_score).
            Best possible score is 1.0, lower values are worse.
            The default is (None, None).
        neg_mean_absolute_error : tuple[float, float], optional
            Mean absolute error regression loss of the experiment
            in the order (train_score, test_score).
            The default is (None, None).
        neg_mean_squared_error : tuple[float, float], optional
            Mean squared error regression loss of the experiment
            in the order (train_score, test_score).
            The default is (None, None).
        r2 : tuple[float, float], optional
            (coefficient of determination) regression score function of the experiment
            in the order (train_score, test_score). Best possible score is 1.0
            and it can be negative (because the model can be arbitrarily worse).
            In the general case when the true y is non-constant, a constant model that always
            predicts the average y disregarding the input features would get a score of 0.0.
            The default is (None, None).
        fit_time : float, optional
            Model fitting time of that experiment. The default is None.
        sort_by : str, optional
            The metric used to sort the output dataframe.
            The default is 'neg_mean_absolute_error'.

    Returns:
        pd.DataFrame
            The dataframe populated with the regression metrics mentioned
            as parameters and sorted by the metric listed in sort_by.

    Examples:
        >>> df_test = regression_model_evaluation_track(model_name='regTest1', feature_count=9, neg_mean_squared_error=(0.77, 0.8), fit_time=1.51235)
        >>> print(df_test) # doctest: +ELLIPSIS
          model_name  feature_count  ... test_r2 train_fit_time
        0   regTest1              9  ...    None         1.5124
        <BLANKLINE>
        [1 rows x 11 columns]

        >>> df_test = regression_model_evaluation_track(\
            eval_metric2_val=df_test.to_dict('list'), model_name='regTest2', feature_count=9, neg_mean_squared_error=(0.9912, 1.22212), fit_time=9.62321\
            )
        >>> print(df_test) # doctest: +ELLIPSIS
          model_name  feature_count  ... test_r2 train_fit_time
        1   regTest2              9  ...    None         9.6232
        0   regTest1              9  ...    None         1.5124
        <BLANKLINE>
        [2 rows x 11 columns]
    """
    if eval_metric2_val is None:
        eval_metric2_val = {
            "model_name": [],
            "feature_count": [],
            "train_explained_variance": [],
            "test_explained_variance": [],
            "train_neg_mean_absolute_error": [],
            "test_neg_mean_absolute_error": [],
            "train_neg_mean_squared_error": [],
            "test_neg_mean_squared_error": [],
            "train_r2": [],
            "test_r2": [],
            "train_fit_time": [],
        }

    eval_metric2_val["model_name"].append(model_name)
    eval_metric2_val["feature_count"].append(feature_count)
    eval_metric2_val["train_explained_variance"].append(explained_variance[0])
    eval_metric2_val["test_explained_variance"].append(explained_variance[1])
    eval_metric2_val["train_neg_mean_absolute_error"].append(neg_mean_absolute_error[0])
    eval_metric2_val["test_neg_mean_absolute_error"].append(neg_mean_absolute_error[1])
    eval_metric2_val["train_neg_mean_squared_error"].append(neg_mean_squared_error[0])
    eval_metric2_val["test_neg_mean_squared_error"].append(neg_mean_squared_error[1])
    eval_metric2_val["train_r2"].append(r2[0])
    eval_metric2_val["test_r2"].append(r2[1])
    eval_metric2_val["train_fit_time"].append(fit_time)

    evaluation_df = pd.DataFrame(eval_metric2_val)

    return evaluation_df.sort_values(by=sort_by, ascending=False).round(4)


######################################


class DropCorrelatedFeatures:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        numeric_features: list,
        categorical_features: list = None,
        ignore_cols: str | list = None,
        round_off: int = 4,
    ):
        """
        Constructor to initialize the class.

        Args:
            dataframe : pd.DataFrame
                The dataframe of interest.
            target_column : str
                The numerical target column.
            numeric_features : list
                The list of possible numerical features in the dataframe.s
            categorical_features : list, optional
                The list of possible categorical features in the dataframe. The default is None.
            ignore_cols : str | list, optional
                The columns to be ignored from correlation analysis. The default is None.
            round_off : int, optional
                Precision of the correlation values. The default is 4.
        """
        self.df = dataframe
        self.tc = target_column
        self.nf = numeric_features
        self.ro = round_off
        self.cf = [] if categorical_features is None else categorical_features

        if ignore_cols is not None:
            if isinstance(ignore_cols, list):
                self.ic = ignore_cols
            elif isinstance(ignore_cols, str):
                self.ic = [ignore_cols]
        else:
            self.ic = []

    @staticmethod
    def cramersV_association(df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to copute the categorical association between the passed categorical
        features using CramersV association.

        Includes bias correction as mentioned in:
            https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

        Args:
            df : pd.DataFrame
                the dataframe of interest with only categorical featuers.

        Returns:
            pd.DataFrame
                dataframe with index and columns are the categorical features
                and values are the CramersV association metric.
        """
        associated_features = []
        nominal_columns_product = list(product(df.columns, df.columns))

        for feat_prod in nominal_columns_product:
            if len(set(feat_prod)) == 1:
                continue
            else:
                crosstab = pd.crosstab(df[feat_prod[0]], df[feat_prod[1]])
                chisq = chi2_contingency(crosstab)
                chi2, pvalue = chisq[0], chisq[1]
                r, k = crosstab.shape
                N = crosstab.sum().sum()
                phi2_corr = max(0, (chi2 / N) - ((k - 1) * (r - 1) / (N - 1)))
                r_correction = r - ((r - 1) ** 2) / (N - 1)
                k_correction = k - ((k - 1) ** 2) / (N - 1)
                min_denom = min(r_correction - 1, k_correction - 1)
                if min_denom == 0:
                    V = np.sqrt(phi2_corr)
                else:
                    V = np.sqrt(phi2_corr / (min(r_correction - 1, k_correction - 1)))

            associated_features.append((feat_prod[0], feat_prod[1], chi2, V, pvalue))

        corr_df = pd.DataFrame(
            [(af[0], af[1], af[3]) for af in associated_features],
            columns=[1, 2, "CorrValue"],
        ).pivot(index=1, columns=2, values="CorrValue")

        return corr_df.fillna(1)

    def num_fs_drop(self, threshold: float = 0.9, verbose: int = 1) -> list:
        """
        Function to inspect high correlation numerical feature pairs and drop according to the
        following example:

        For any pair of columns (f_i, f_j) where f_i appears before f_j in the ordering
        of the columns of input df, and corr(f_i, f_j) >= corr_threshold:
            If corr(f_i, target_var) <= corr(f_j, target_var), then drop f_i.
            Otherwise drop f_j

        NOTE: This approach might produce different results depending on the order
        of columns in the dataframe.

        Args:
            threshold : float, optional
                The correlation threshold beyond which to consider dropping columns.
                The default is 0.9.
            verbose : int, optional
                Indicator to inspect mid-way steps. The default is 1.

        Returns:
            list
                list of numerical features that can be safely removed.
        """
        num_feats_to_drop = []
        target_vals = self.df[self.tc]
        corr_df = self.df[self.nf].corr().abs().round(self.ro)

        upper_triangle = corr_df.where(
            np.triu(np.ones(corr_df.shape), k=1).astype(bool)
        )
        upper_triangle_filtered = upper_triangle[upper_triangle >= threshold]
        corr_pairs_unstacked = upper_triangle_filtered.unstack()[
            upper_triangle_filtered.unstack().notna()
        ]

        for feature_pair, corr_value in corr_pairs_unstacked.items():
            do_not_check = [self.tc] + num_feats_to_drop + self.ic
            if verbose == 2:
                print(f"Ignoring: {do_not_check} columns...")

            if any(feature in feature_pair for feature in do_not_check):
                continue

            f1, f2 = feature_pair
            corr_f1_w_target = self.df[f1].corr(target_vals)
            corr_f2_w_target = self.df[f2].corr(target_vals)
            if verbose == 1:
                print(f"{feature_pair} has a corr. value = {corr_value}")
                print(
                    f"\t{f1} has a correlation of {corr_f1_w_target} with the target column {self.tc}"
                )
                print(
                    f"\t{f2} has a correlation of {corr_f2_w_target} with the target column {self.tc}"
                )

            if abs(corr_f1_w_target) <= abs(corr_f2_w_target):
                if verbose:
                    print(f"\t\tDropping {f1}")
                num_feats_to_drop.append(f1)

            elif abs(corr_f1_w_target) > abs(corr_f2_w_target):
                if verbose:
                    print(f"\t\tDropping {f2}")
                num_feats_to_drop.append(f2)

            print(
                "-------------------------------------------------------------------------------------------"
            )
            print()

        return num_feats_to_drop

    def cat_fs_drop(
        self,
        threshold: float = 0.9,
        verbose: int = 1,
        strategy: str = "mean_target_corr",
        save_corr_file: bool = True,
        path: str = None,
    ) -> list:
        """
        Function to inspect high association categorical feature pairs and drop according to the
        following example:
            For any pair of columns (f_i, f_j) where f_i appears before f_j in the ordering
            of the columns of input df, and corr(f_i, f_j) >= corr_threshold:
                If mean(corr(f_i)) >= mean(corr(f_j), then drop f_i.
                Otherwise drop f_j

        Parameters:
            threshold : float, optional
                The correlation threshold beyond which to consider dropping columns.
                The default is 0.9.
            verbose : int, optional
                Indicator to inspect mid-way steps. The default is 1.
            strategy : str, optional
                Either to remove by considering the mean correlation of a variable
                or to consider the correlation with target.
                The default is "mean_target_corr".
            save_corr_file : bool, optional
                Whether to save the cramersV correlation file. The default is True.
            path : str, optional
                Path to save the cramersV correlation file. The default is None.

        Returns:
            list
                list of categorical features that can be safely removed.
        """
        cat_feats_to_drop = []
        use_cols = list(set(self.cf) - set(self.ic))
        corr_df = DropCorrelatedFeatures.cramersV_association(self.df[use_cols])

        if save_corr_file:
            if path is None:
                path = "CramersV_Association.csv"
            corr_df.to_csv(path, index=False)

        upper_triangle = corr_df.where(
            np.triu(np.ones(corr_df.shape), k=1).astype(bool)
        )
        upper_triangle_filtered = upper_triangle[upper_triangle >= threshold]
        corr_pairs_unstacked = upper_triangle_filtered.unstack()[
            upper_triangle_filtered.unstack().notna()
        ]

        corr_pairs_unstacked_df = pd.DataFrame(corr_pairs_unstacked).reset_index()
        corr_pairs_unstacked_df.columns = ["f1", "f2", "CramV"]
        if verbose == 1:
            print(corr_pairs_unstacked_df)

        for feature_pair, corr_val in corr_pairs_unstacked.items():
            do_not_check = self.ic + cat_feats_to_drop
            if verbose == 2:
                print(f"Ignoring: {do_not_check} columns...")
            if any(feature in feature_pair for feature in do_not_check):
                continue

            f1, f2 = feature_pair
            print(f"For {f1} {f2}")
            val1 = corr_pairs_unstacked_df[corr_pairs_unstacked_df["f1"] == f1]["CramV"]
            val2 = corr_pairs_unstacked_df[corr_pairs_unstacked_df["f2"] == f1]["CramV"]
            mean_corr_f1 = (val1.sum() + val2.sum()) / (len(val1) + len(val2))

            val1 = corr_pairs_unstacked_df[corr_pairs_unstacked_df["f1"] == f2]["CramV"]
            val2 = corr_pairs_unstacked_df[corr_pairs_unstacked_df["f2"] == f2]["CramV"]
            mean_corr_f2 = (val1.sum() + val2.sum()) / (len(val1) + len(val2))

            if verbose == 1:
                print(f"{feature_pair} has a corr. value = {corr_val}")
                print(f"\t{f1} has a mean correlation of {mean_corr_f1}")
                print(f"\t{f2} has a mean correlation of {mean_corr_f2}")

            if mean_corr_f1 >= mean_corr_f2:
                if verbose:
                    print(f"Dropping: {f1}")
                cat_feats_to_drop.append(f1)

            else:
                if verbose:
                    print(f"Dropping: {f2}")
                cat_feats_to_drop.append(f2)

        return cat_feats_to_drop


######################################

def get_permutation_importance_fs(
    fitted_estimator,
    X_test_df: pd.DataFrame,
    y_test: np.array,
    scoring: str = None,
    n_repeats: int = 25,
    n_jobs: int = -1,
    seed: int = 89,
    top_n_features: int = 10,
    plot: bool = True,
    figsize: tuple[float, float] = (15, 7.5),
    color: str = "g",
    error_lw: float = 1.5,
    savefig_path: str = None,
    matplotlib_style: str = "fivethirtyeight",
):
    """
    Function to compute the permutation feature importance using the testing
    dataframes.

    Args:
        fitted_estimator : TYPE
            Pipeline type or model type object to test on X_test_df and y_test.
        X_test_df : pd.DataFrame
            Dataframe with independent variables.
        y_test : np.array
            The dependent variable, as array.
        scoring : str
            The performance metric to use to compute feature importances.
        n_repeats : int, optional
            The number of times columns should be shuffled. The default is 25.
        n_jobs : int, optional
            Threading. The default is -1.
        seed : int, optional
            Random seed. The default is 89.
        top_n_features : int, optional
            Top 'n' features required. The default is 10.
        plot : bool, optional
            Indicator to plot the most important features. The default is True.
        figsize : tuple(float, float), optional
            Size of the figure. The default is (15, 7.5).
        color : str, optional
            Color of the bar charts in the figure. The default is "g".
        error_lw : float, optional
            Linewidth of error bars. The default is 1.5.
        savefig_path : str, optional
            Path to save the figure. The default is None.
        matplotlib_style : str, optional
            Matplotlib style. The default is "fivethirtyeight".

    Returns:
        top_imp_features : TYPE
            The top 'n' most important features that influence the given performance
            metric.
    """
    # Sklearn permutation importance function
    perm_imp = permutation_importance(
        estimator=fitted_estimator,
        X=X_test_df,
        y=y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        n_jobs=n_jobs,
        random_state=seed,
    )

    # Getting column names from the X dataframe
    feature_names = X_test_df.columns

    # Sorting the permutation importance values in descending order
    perm_imp_sorted_idx = perm_imp.importances_mean.argsort()[::-1]
    # Extracting the top_n features from the perm_imp object
    top_imp_features = np.array(feature_names)[perm_imp_sorted_idx][:top_n_features]

    if plot:
        plt.style.use(matplotlib_style)
        x = top_imp_features[::-1]
        x_error = perm_imp.importances_std[perm_imp_sorted_idx][:top_n_features][::-1]
        y = np.array(perm_imp.importances_mean)[perm_imp_sorted_idx][:top_n_features][
            ::-1
        ]
        fig = plt.figure(figsize=figsize)
        plt.barh(
            x,
            y,
            xerr=x_error,
            color=color,
            error_kw=dict(lw=error_lw, capsize=2.5, capthick=2.5),
        )
        plt.title(f'Permutation Importance: TOP-{top_n_features} features')
        plt.show()
        if savefig_path is not None:
            fig.savefig(savefig_path)

    return top_imp_features


######################################

def roc_auc_w_cv(
    classifier, X: np.ndarray, y: np.ndarray, cv, cv_split, pos_label: str
):
    """
    Function to plot the ROC curves for each cross validation folds.
    
    ROC curves typically feature true positive rate (TPR) on the Y axis, and
    false positive rate (FPR) on the X axis. This means that the top left corner
    of the plot is the “ideal” point - a FPR of zero, and a TPR of one.

    This is not very realistic, but it does mean that a larger Area Under the Curve (AUC)
    is usually better. The “steepness” of ROC curves is also important, since it is ideal
    to maximize the TPR while minimizing the FPR.

    Args:
        classifier : 
            The model to be used for training..
        X : np.ndarray
            numpy array of features.
        y : np.ndarray
            numpy array of target.
        cv : 
            cross validation object.
        cv_split : 
            For certain cv objects, the technique to splits maybe different, so
            explicitly pass.
        pos_label : str
            The name of the class treated as positive.

    Returns:
        None.
        
    Examples:
        This example shows the ROC response of different datasets, created from
        K-fold cross-validation. Taking all of these curves, it is possible to 
        calculate the mean AUC, and see the variance of the curve when the training 
        set is split into different subsets. This roughly shows how the classifier 
        output is affected by changes in the training data, and how different the 
        splits generated by K-fold cross-validation are from one another.
        
        Run the following code to test the function:
            import numpy as np
            from sklearn.datasets import load_iris
            
            iris = load_iris()
            target_names = iris.target_names
            X, y = iris.data, iris.target
            X, y = X[y != 2], y[y != 2]
            n_samples, n_features = X.shape
            random_state = np.random.RandomState(0)
            X = np.concatenate([X, random_state.randn(n_samples, 2500 * n_features)], axis=1)

            # Setting up cv
            from sklearn.model_selection import StratifiedKFold
            from sklearn.linear_model import LogisticRegression
            
            n_splits = 5
            cv = StratifiedKFold(n_splits=n_splits)
            clf = LogisticRegression(solver='liblinear')
            roc_auc_w_cv(clf, X, y, cv=cv, cv_split=cv.split(X, y), pos_label='versicolor(label=1)')
    """
    # Tracking true positive rates, AUC scores
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(12, 9))

    for fold, (train, test) in enumerate(cv_split):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier, X, y, name=f"ROC fold: {fold}", alpha=0.4, lw=1, ax=ax
        )
        # Notice that you have to pass in:
        #     A set of points where you want the interpolated value (mean_fpr)
        #     A set of points with a known value (viz.fpr)
        #     The set of known values (viz.tpr)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # Plotting the chance curve
    ax.plot([0, 1], [0, 1], "k--", label="Chance: (AUC=0.5)", linewidth=1.2)

    # Plotting the mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.75,
    )

    # Plotting the confidence interval around the mean ROC
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability\n(Positive label: {pos_label})",
    )
    ax.legend(loc="lower right")
    plt.show()
    
    
def synthetic_data_generate(
    distribution: str,
    param1: float,
    param2: float = None,
    seed: int = 23,
    no_samples: int = 2500,
    plot_data: bool = True,
    color: str = 'cyan',
    bins: int = 50,
    figsize: tuple[int, int] = (10, 8)
) -> np.array:
    """
    Function to generate synthetic data for 'normal', 'uniform', 'exponential', 'lognormal',
    'chisquare' and 'beta' distributions.

    Args:
        distribution : str
            The distribution for which the synthetic data is to be created.
        param1 : float
            First parameter of the distribution. Conforms to np.random.'distribution'.
        param2 : float
            Second parameter of the distribution. Conforms to np.random.'distribution'.
        seed : int, optional
            Seed for data generation. The default is 23.
        no_samples : int, optional
            Number of samples that need to be generated. The default is 2500.
        plot_data : bool, optional
            Flat to plot the generated data. The default is True.
        color : str, optional
            Color to be used in the plot. The default is 'cyan'.
        bins : int, optional
            Number of bins in the histogram. The default is 50.
        figsize : TYPE, optional
            The size of the figure. The default is (10, 8).

    Raises:
        NameError: If the distribution string passed doesn't match the
        allowed distribution name strings.

    Returns:
        synthetic_data_dist : np.array
            Synthetic data that belongs to the passed distribution.

    Examples:
        >>> print(synthetic_data_generate('beta', 0.1 ,0.7, no_samples=25))
        Evaluating: np.random.beta(0.1, 0.7, 25)
        [1.48106312e-03 2.95996514e-01 4.76909262e-07 6.47296485e-08
         2.80635484e-02 9.87265825e-27 6.21458267e-01 5.20839780e-03
         9.02038101e-01 2.93009394e-05 2.16573885e-01 1.29939222e-05
         9.00048607e-01 8.05760306e-04 4.53939206e-01 1.97057215e-01
         1.21454052e-09 5.22063615e-08 3.20164980e-01 2.94227502e-08
         7.13676027e-03 3.27952428e-02 2.47818967e-07 4.10903462e-03
         7.37451142e-04]

        >>> print(synthetic_data_generate('exponential', 1, no_samples=15))
        Evaluating: np.random.exponential(1, 15)
        [7.28355552e-01 2.93675803e+00 1.45012810e+00 3.31837177e-01
         2.49802467e-01 1.15906982e+00 1.82888761e-01 4.98308403e-01
         9.62471714e-01 5.30909452e-01 2.46792402e-03 2.15444256e+00
         2.16236707e+00 3.57260386e-01 8.90578798e-01]
    """

    allowed_dists = ['normal', 'uniform', 'exponential', 'lognormal', 'chisquare', 'beta']

    if distribution not in allowed_dists:
        raise NameError(
            f"{distribution} doesn't match any of the allowed distributions {allowed_dists}."
            )

    np.random.seed(seed=seed)
    if param2:
        evaluation_string = f'np.random.{distribution}({param1}, {param2}, {no_samples})'
    else:
        evaluation_string = f'np.random.{distribution}({param1}, {no_samples})'

    print(f"Evaluating: {evaluation_string}")
    synthetic_data_dist = eval(evaluation_string)
    if plot_data:
        _, ax = plt.subplots(figsize=figsize)
        ax.hist(synthetic_data_dist, bins=bins, color=color)
        plt.title(f"Synthetic {distribution} distribution")
        plt.show()

    return synthetic_data_dist

##############################

######################################


if __name__ == "__main__":
    import doctest
    doctest.testmod()

######################################
