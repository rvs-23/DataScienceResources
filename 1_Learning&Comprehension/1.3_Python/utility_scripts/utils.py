import itertools
from functools import reduce

from typing import Optional

import numpy as np
import pandas as pd


def get_df_info(
    df: pd.DataFrame, subset: Optional[list] = None, num_attrs_info: bool = False
) -> pd.DataFrame:
    """
    Function to get feature level information from a dataframe.

    Args:
        df : pd.DataFrame
            The pandas dataframe of interest.
        subset : Optional[list], optional
            If a subset of columns is to be used. The default is None.
        num_attrs_info : bool, optional
            True, if min, max, median, skewness, kurtosis values is required for numerical colums.
            The default is False.

    Returns:
        pd.DataFrame
            Dataframe with all necessary information.
    """
    if subset is None:
        columns = list(df.columns)
    else:
        columns = subset

    rows_len, cols_len = df.shape

    column_type_python = [
        type(df[df[column].notna()][column].sample().to_numpy()[0])
        for column in columns
    ]
    column_type_pandas = [df[column].dtype for column in columns]

    null_count = [df[column].isna().sum() for column in columns]
    null_percent = [(count / rows_len) * 100 for count in null_count]

    unique = [df[column].unique() for column in columns]
    unique_count = [len(unique_vals) for unique_vals in unique]
    unique_percent = [(count / rows_len) * 100 for count in unique_count]

    mode_values = [list(df[column].mode().to_numpy()) for column in columns]

    df_info_dict = {
        "Column": columns,
        "Dtype_pandas": column_type_pandas,
        "Dtype_python": column_type_python,
        "Null_count": null_count,
        "Null%": null_percent,
        "Unique_count": unique_count,
        "Unique%": unique_percent,
        "Mode": mode_values,
        "Unique_values": unique,
    }

    if num_attrs_info:
        # Numerical specific information
        numeric_columns = [
            column
            for column in columns
            if df[column].dtype not in ["O", str, "datetime64[ns]"]
        ]

        #  Attributes: Min, Max, Median, Skewness, Kurtosis
        min_values = [
            df[column].min() if column in numeric_columns else None
            for column in columns
        ]
        max_values = [
            df[column].max() if column in numeric_columns else None
            for column in columns
        ]
        median_values = [
            df[column].median() if column in numeric_columns else None
            for column in columns
        ]
        skewness_values = [
            df[column].skew() if column in numeric_columns else None
            for column in columns
        ]
        kurtosis_values = [
            df[column].kurtosis() if column in numeric_columns else None
            for column in columns
        ]

        df_info_dict["Min"] = min_values
        df_info_dict["Max"] = max_values
        df_info_dict["Median"] = median_values
        df_info_dict["Skewness"] = skewness_values
        df_info_dict["Kurtosis"] = kurtosis_values

    return pd.DataFrame(df_info_dict)


##############################


def get_common_elements(*elements: list | tuple | set) -> set:
    """
    Helper Function to get common elements between two or more sequences.
    Used by dataframe_common_columns()

    NOTE: Doesn't ignore np.nan

    Args:
        *elements : list|tuple|set
            Collection of elements to be considered.

    Returns:
        set
            Set of all elements common to all.

    """
    return set(reduce(set.intersection, map(set, elements)))


##############################


def common_cols_by_name_bw_dfs(df_dictionary: dict, comb_size: int = 2) -> dict:
    """
    Function to get common column NAMES among 2 or more dataframes.

    Args:
        df_dictionary : dict
            Dataframe dictionary with key as required name and values as pandas df.
        comb_size : int, optional
            Number of combinations of df taken at a time. The default is 2.

    Returns:
        dict
            Dictionary -> Key: df combinations, Values: common columns
    """

    # To store the connections result
    connections_result_dict = {}

    for comb_of_df_names in itertools.combinations(df_dictionary.keys(), comb_size):
        # Using the helper function to get the common column names
        # comb_of_df_names is a tuple of df combinations.
        # Creating a list of list with elements as the df columns.
        # Unpacking the outside list to pass a bunch of list of df columns to the helper.
        common_cols = get_common_elements(
            *[df_dictionary[comb_of_df_names[i]].columns for i in range(comb_size)]
        )

        # Add to connections dict if common cols are non-empty
        if common_cols:
            connections_result_dict[comb_of_df_names] = common_cols

    return connections_result_dict


##############################


def find_const_and_null_cols_df(
    df: pd.DataFrame, verbose: int = 0, ignore_cols: Optional[list] = None
) -> list:
    """
    Function to find from a dataframe all the columns with only constant values
    or only null values.

    Args:
        df : pd.DataFrame
            The required dataframe.
        verbose : int, optional
            Set 1 to print the null and constant columns seperately.
            The default is 0.
        ignore_cols : Optional[list], optional
            Columns to be ignored from consideration.
            The default is None.

    Returns:
        list
            List of columns found with all constant + all null values.

    """

    df_nuniq = df.nunique()

    const_cols = list(df.columns[df_nuniq == 1])
    all_null_cols = list(df.columns[df_nuniq == 0])

    if ignore_cols:
        const_cols = list(set(const_cols) - set(ignore_cols))
        all_null_cols = list(set(all_null_cols) - set(ignore_cols))

    if verbose:
        print(f"Constant columns: \n\t{const_cols}")
        print(f"All Null Columns: \n\t{all_null_cols}")

    return const_cols + all_null_cols
