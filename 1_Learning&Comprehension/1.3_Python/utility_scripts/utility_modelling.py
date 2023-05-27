"""
Module that contains utility functions for Modelling:
    1. classification_model_evaluation_track: Function to track performance metrics
    for classification experiments.
    2. regression_model_evaluation_track: Function to track performance metrics
    for regression experiments.
    3. pearson_corr_drop_features: Function to inspect high correlation feature pairs and
    drop according to a strategy.
"""

import numpy as np
import pandas as pd


def classification_model_evaluation_track(
        eval_metric2_val: dict = None,
        model_name: str = None,
        feature_count: int = None,
        accuracy_score: tuple[float, float] = (None, None),
        recall_score: tuple[float, float] = (None, None),
        bal_accuracy_score: tuple[float, float] = (None, None),
        f1_score: tuple[float, float] = (None, None),
        roc_auc: tuple[float, float] = (None, None),
        fit_time: float = None,
        sort_by: str = 'test_accuracy'
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
        >>> df_test = classification_model_evaluation_track(\
            model_name='test1', feature_count=9, recall_score=(.99, .85), roc_auc=(0.56721, 0.55116), fit_time=11.2\
            )
        >>> print(df_test) # doctest: +ELLIPSIS
          model_name  feature_count  ... test_roc_auc train_fit_time
        0      test1              9  ...       0.5512           11.2
        <BLANKLINE>
        [1 rows x 13 columns]

        >>> df_test = classification_model_evaluation_track(\
            eval_metric2_val=df_test.to_dict('list'), model_name='test2', feature_count=12, accuracy_score=(.911, .886), recall_score=(0.2341, 0.3121), fit_time=5.2\
            )
        >>> print(df_test) # doctest: +ELLIPSIS
          model_name  feature_count  ...  test_roc_auc  train_fit_time
        1      test2             12  ...           NaN             5.2
        0      test1              9  ...        0.5512            11.2
        <BLANKLINE>
        [2 rows x 13 columns]
    """

    if eval_metric2_val is None:
        eval_metric2_val = {
            'model_name': [],
            'feature_count': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'train_recall': [],
            'test_recall': [],
            'train_bal_accuracy': [],
            'test_bal_accuracy': [],
            'train_f1_score': [],
            'test_f1_score': [],
            'train_roc_auc': [],
            'test_roc_auc': [],
            'train_fit_time': []
            }

    eval_metric2_val['model_name'].append(model_name)
    eval_metric2_val['feature_count'].append(feature_count)
    eval_metric2_val['train_accuracy'].append(accuracy_score[0])
    eval_metric2_val['test_accuracy'].append(accuracy_score[1])
    eval_metric2_val['train_recall'].append(recall_score[0])
    eval_metric2_val['test_recall'].append(recall_score[1])
    eval_metric2_val['train_bal_accuracy'].append(bal_accuracy_score[0])
    eval_metric2_val['test_bal_accuracy'].append(bal_accuracy_score[1])
    eval_metric2_val['train_f1_score'].append(f1_score[0])
    eval_metric2_val['test_f1_score'].append(f1_score[1])
    eval_metric2_val['train_roc_auc'].append(roc_auc[0])
    eval_metric2_val['test_roc_auc'].append(roc_auc[1])
    eval_metric2_val['train_fit_time'].append(fit_time)

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
        sort_by: str = 'train_neg_mean_squared_error'
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
            The metric to be used to sort the output dataframe.
            The default is 'neg_mean_absolute_error'.

    Returns:
        pd.DataFrame
            The dataframe populated with the regression metrics mentioned
            as parameters and sorted by the metric listed in the sort_by parameter.

    Examples:
        >>> df_test = regression_model_evaluation_track(\
            model_name='regTest1', feature_count=9, neg_mean_squared_error=(0.77, 0.8), fit_time=1.51235\
            )
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
            'model_name': [],
            'feature_count': [],
            'train_explained_variance': [],
            'test_explained_variance': [],
            'train_neg_mean_absolute_error': [],
            'test_neg_mean_absolute_error': [],
            'train_neg_mean_squared_error': [],
            'test_neg_mean_squared_error': [],
            'train_r2': [],
            'test_r2': [],
            'train_fit_time': []
            }

    eval_metric2_val['model_name'].append(model_name)
    eval_metric2_val['feature_count'].append(feature_count)
    eval_metric2_val['train_explained_variance'].append(explained_variance[0])
    eval_metric2_val['test_explained_variance'].append(explained_variance[1])
    eval_metric2_val['train_neg_mean_absolute_error'].append(neg_mean_absolute_error[0])
    eval_metric2_val['test_neg_mean_absolute_error'].append(neg_mean_absolute_error[1])
    eval_metric2_val['train_neg_mean_squared_error'].append(neg_mean_squared_error[0])
    eval_metric2_val['test_neg_mean_squared_error'].append(neg_mean_squared_error[1])
    eval_metric2_val['train_r2'].append(r2[0])
    eval_metric2_val['test_r2'].append(r2[1])
    eval_metric2_val['train_fit_time'].append(fit_time)

    evaluation_df = pd.DataFrame(eval_metric2_val)

    return evaluation_df.sort_values(by=sort_by, ascending=False).round(4)

######################################


def pearson_corr_drop_features(
        df: pd.DataFrame,
        target_col: str,
        ignore_cols: list = None,
        corr_thresh: float = 0.9,
        round_off: int = 5,
        verbose: bool= True
) -> list:
    """
    Function to inspect high correlation feature pairs and drop according to the
    following strategy:

    corr(f1, f2) = 0.95
    if corr(f1, target_var) < corr(f2, target_var)
    drop f1

    NOTE: This approach might produce different results depending on the order
    of columns in the dataframe.

    Args:
        df : pd.DataFrame
            The dataframe with NUMERICAL features.
        target_col : str
            The target in the dataset.
        ignore_cols : list, optional
            The list of columns to ignore from the correlation analysis.
            The default is None.
        corr_thresh : float, optional
            The threshold beyond which to consider correlation pairs.
            The default is 0.9.
        round_off : int, optional
            Round off decimal places for correlation values. The default is 5.
        verbose : bool, optional
            Flag to print the pairs and the features being dropped from those pairs.
            The default is True.

    Returns:
        list
            list of features with correlation beyond the threshold and minimum
            correlation with the target.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(89)
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=3, n_informative=2, n_targets=1, random_state=89)
        >>> df_reg_test = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(3)])
        >>> df_reg_test['high_corr_wx_2'] = df_reg_test['feat_2']*2 + np.random.random(df_reg_test.shape[0])
        >>> df_reg_test['high_corr_wx_4'] = df_reg_test['feat_1']*3 + np.random.random(df_reg_test.shape[0])
        >>> print(pearson_corr_drop_features(df_reg_test, target_col='feat_0', corr_thresh=0.9)) # doctest: +NORMALIZE_WHITESPACE
        ('high_corr_wx_2', 'feat_2') has a corr. value = 0.98834
        	high_corr_wx_2 has a correlation of -0.06162 with the target column feat_0
        	feat_2 has a correlation of -0.03578 with the target column feat_0
        		Dropping feat_2
        -------------------------------------------------------------------------------------------
        <BLANKLINE>
        ('high_corr_wx_4', 'feat_1') has a corr. value = 0.99502
        	high_corr_wx_4 has a correlation of 0.10857 with the target column feat_0
        	feat_1 has a correlation of 0.10966 with the target column feat_0
        		Dropping high_corr_wx_4
        -------------------------------------------------------------------------------------------
        <BLANKLINE>
        ['feat_2', 'high_corr_wx_4']
    """

    feats_to_drop = []
    if ignore_cols is None:
        ignore_cols = []

    dataframe = df.copy()
    target = dataframe[target_col]
    dataframe.drop(columns=target_col, inplace=True)

    # Correlation with absolute values.
    df_corr = dataframe.corr().abs().round(round_off)

    # Select upper triangle of correlation matrix.
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))

    # Unstack upper triangle, filter non-null values and choose pairs beyond the threshold.
    corr_pairs_unstacked_filt = upper.unstack()[upper.unstack().notna()]
    corr_pairs_unstacked_filt = corr_pairs_unstacked_filt[corr_pairs_unstacked_filt>=corr_thresh]

    for feat_pair in corr_pairs_unstacked_filt.index:
        feat1, feat2 = feat_pair

        # Ignore if the columns should be ignored
        if (feat1 in ignore_cols) or (feat2 in ignore_cols):
            continue

        # Ignore if the feature is already in the feats_to_drop list
        if (feat1 in feats_to_drop) or (feat2 in feats_to_drop):
            continue

        # Find and check the correlation with target is smaller for which feature
        corr_val = corr_pairs_unstacked_filt[feat_pair]
        corr_tar_w_feat1, corr_tar_w_feat2 = round(dataframe[feat1].corr(target), round_off), round(dataframe[feat2].corr(target), round_off)
        if verbose:
            print(f"{feat_pair} has a corr. value = {corr_val}")
            print(f"\t{feat1} has a correlation of {corr_tar_w_feat1} with the target column {target_col}")
            print(f"\t{feat2} has a correlation of {corr_tar_w_feat2} with the target column {target_col}")

        if abs(corr_tar_w_feat1) <= abs(corr_tar_w_feat2):
            print(f"\t\tDropping {feat1}")
            feats_to_drop.append(feat1)
        elif abs(corr_tar_w_feat1) > abs(corr_tar_w_feat2):
            print(f"\t\tDropping {feat2}")
            feats_to_drop.append(feat2)

        if verbose:
            print('-------------------------------------------------------------------------------------------')
            print()

    return feats_to_drop

######################################


if __name__ == "__main__":
    import doctest
    doctest.testmod()

######################################
