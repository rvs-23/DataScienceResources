"""
Module that contains general utility functions.
    1. get_df_info: Function to get information about a given dataset's columns.
    2. dataframe_common_columns: Function to extract common dataframe COLUMN NAMES
    between 2 or more dataframes.
    3. date_info_extract: Function to extract all date related information from a datetime column.
    4. get_sign_info_as_percent: Function to get the % of +ves, -ves and zeros.
    present in a set of features within a dataframe.
    5. find_const_and_null_cols: Function to drop columns with all constant or all null values.
    6. synthetic_data_generate: Function to generate synthetic data for 'normal', 'uniform',
    'exponential', 'lognormal', 'chisquare' and 'beta' distributions.
"""

import itertools
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_df_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to get information about a given dataset's columns.

    Args:
        df : pd.DataFrame
            The pandas dataframe of interest.

    Returns:
        pd.DataFrame: The dataframe with column names, column type, null count,
        null %, unique count and unique values as columns.

    Examples:
        >>> df = pd.DataFrame({'cat_col': ['A', 'B', np.nan], 'num_col': [4, -1.4, 2.22]})
        >>> print(get_df_info(df)) # doctest: +ELLIPSIS
            column              column_type  ...  unique_percent      unique_values
        0  cat_col            <class 'str'>  ...       66.666667        [A, B, nan]
        1  num_col  <class 'numpy.float64'>  ...      100.000000  [4.0, -1.4, 2.22]
        <BLANKLINE>
        [2 rows x 7 columns]
"""
    columns = list(df.columns)
    column_types = [type(cell) for cell in df.iloc[np.random.randint(1, len(df)-1), :]]
    null_count = [df[col].isna().sum() for col in df.columns]
    null_percent = [(df[col].isna().sum()/df.shape[0])*100 for col in df.columns]
    unique_count = [df[col].nunique() for col in df.columns]
    unique_percent = [(df[col].nunique()/df.shape[0])*100 for col in df.columns]
    unique = [df[col].unique() for col in df.columns]

    return pd.DataFrame(
        {
            'column': columns,
            'column_type': column_types,
            'null_count': null_count,
            'null_percent': null_percent,
            'unique_count': unique_count,
            'unique_percent': unique_percent,
            'unique_values': unique
        }
    )

##############################


def common_elements(*elements) -> list:
    """
    Helper Function to get common elements between two or more sequences.

    Args:
        *elements: Sequence type

    Returns:
        list: list of elements common to all input sequence types

    Examples:
        >>> common_elements([1, 2, 2, 3, 4, 5], [2, 7, 9])
        [2]

        >>> common_elements(('a', 'z', -1, 'c'), ('z', 4, -1))
        ['z', -1]
    """
    return list(reduce(set.intersection, map(set, elements)))


def dataframe_common_columns(
    df_dictionary: dict,
    comb_size: int = 2,
    arrange_by_na: bool = True,
    names_as_multiindex: bool = True,
    possible_keys: list = None
) -> pd.DataFrame:
    """
    Function to extract common dataframe COLUMN NAMES between 2 or more dataframes.

    Args:
        df_dictionary : dict
            Dictionary with key as dataframe name and value as the dataframe.
        comb_size : int, optional
            To specify how many dataframes are taken at a time. The default is 2.
        arrange_by_na : bool, optional
            Whether the column combinations from dataframes should be ordered by non-null
            combinations. (See Examples, where (b, None) is the first set of commons.
            The default is True.
        names_as_multiindex : bool, optional
            Whether the dataframe names should be made a multi index or kept as
            a tuple index. The default is True.

    Returns:
        df_common : pd.DataFrame
            Dataframe names as index and the common column names as values.

    Examples:
        >>> df1 = pd.DataFrame(data=[[1, 2], [3, 4]], columns=['a', 'b'])
        >>> df2 = pd.DataFrame(data=[[1, 4], [0, 7]], columns=['l', 'z'])
        >>> df3 = pd.DataFrame(data=[[-1, 1, 8], [0, 4, 3], [0, 0, 4]], columns=['z', 'b', 'l'])
        >>> print(dataframe_common_columns({'df1_name': df1, 'df2_name': df2, 'df3_name': df3}))
                           0     1
        df1_name df3_name  b  None
        df2_name df3_name  z     l

    """

    # To store the connections result
    connections_result_dict = {}
    for comb_of_df_names in itertools.combinations(df_dictionary.keys(), comb_size):
        # Using the helper function to get the common column names
        common_cols = common_elements(
            *[df_dictionary[comb_of_df_names[i]].columns for i in range(comb_size)]
            )

        # If the user passes a list of possible keys, check only within those
        if possible_keys:
            common_cols = [cols_poss for cols_poss in common_cols if cols_poss in possible_keys]

        # If common cols are non empty
        if common_cols:
            connections_result_dict[comb_of_df_names] = common_cols

    df_common = pd.DataFrame.from_dict(connections_result_dict, orient='index')

    # Arranging by non null column names
    if arrange_by_na:
        # Therefore, using count to order the dataframe because
        # it ignores Null(None) values by default.
        df_common['cols_without_None'] = df_common.count(axis=1)
        df_common.sort_values(
            by='cols_without_None', ascending=True, inplace=True
            )
        df_common.drop(columns=['cols_without_None'], inplace=True)

    # Creating the dataframe names as index
    if names_as_multiindex:
        df_common.index = pd.MultiIndex.from_tuples(df_common.index)

    return df_common

##############################


def date_info_extract(
    df: pd.DataFrame,
    date_column: str,
    year: bool = True,
    month: bool = True,
    month_name: bool = False,
    day: bool = False,
    dayofweek: bool = False,
    week: bool = False,
    quarter: bool = False,
    hour: bool = False,
    minute: bool = False,
    second: bool = False
) -> pd.DataFrame:
    """
    Function to extract all date related information from a datetime column.

    Args:
        df : pd.DataFrame
            The dataframe whose date information needs to be extracted.
        date_column : str
            The datetime type column of interest. The default is None.
        year : bool, optional
            Extract year from the datetime object. The default is True.
        month : bool, optional
            Extract month from the datetime object. The default is True.
        month_name : bool, optional
            Extract month name from the datetime object. The default is False.
        day : bool, optional
            Extract day from the datetime object. The default is False.
        dayofweek : bool, optional
            Extract day of week from the datetime object. The default is False.
        week : bool, optional
            Extract week from the datetime object. The default is False.
        quarter : bool, optional
            Extract quarter from the datetime object. The default is False.
        hour : bool, optional
            Extract hour from the datetime object. The default is False.
        minute : bool, optional
            Extract minute from the datetime object. The default is False.
        second : bool, optional
            Extract second from the datetime object. The default is False.

    Returns:
        df : pd.DataFrame
            Final DataFrame after appending all the truthy datetypes.

    Examples:
        >>> df = pd.DataFrame({'DOB': {0: '6-1-2001', 1: '23-9-1998'}})
        >>> df['DOB'] = pd.to_datetime(df['DOB'], format='%d-%m-%Y')
        >>> print(date_info_extract(df, 'DOB', day=True)) # doctest: +ELLIPSIS
                 DOB  DOB_Year  DOB_Month  DOB_Day
        0 2001-01-06      2001          1        6
        1 1998-09-23      1998          9       23
    """
    # Dictionary to store all the truthy and falsey values.
    extraction_type2bool = {
        'Year': year,
        'Month': month,
        'Month_name': month_name,
        'Day': day,
        'DayOfWeek': dayofweek,
        'Week': week,
        'Quarter': quarter,
        'Hour': hour,
        'Minute': minute,
        'Second': second
    }

    # Filter only truthy
    args_true_filtered = {
        key: val for key, val in extraction_type2bool.items() if val
        }

    for funcs in args_true_filtered:
        df[f"{date_column}_{funcs}"] = eval(f'df[date_column].dt.{funcs.lower()}')

    return df

##############################


def get_sign_info_as_percent(df: pd.DataFrame, *features, print_signed_pcts_col: bool = False):
    """
    Function to get the % of +ves, -ves and zeros present in a set of features
    within a dataframe.

    Args:
        df: pd.DataFrame
            The dataframe with features of interest.
        *features:
            The features of interest.
        print_signed_pcts_col: bool
            Boolean to inspect the % of signed values for that column.
            The default is False.

    Returns:
        df: pd.DataFrame
            Final dataframe with new columns containing the signs(+ve, -ve or 0) of the feature set.

    Examples:
        >>> df = pd.DataFrame(data=[[2, -1, 4], [0, 0, -2], [0, -10, 4], [11, 0, 41], [17, 1, 4]], columns=['a', 'b', 'c'])
        >>> print(get_sign_info_as_percent(df, 'a', 'b', print_signed_pcts_col=True))
        1    60.0
        0    40.0
        Name: a_sign_%, dtype: float64
        <BLANKLINE>
        -1    40.0
         0    40.0
         1    20.0
        Name: b_sign_%, dtype: float64
        <BLANKLINE>
            a   b   c  a_sign_%  b_sign_%
        0   2  -1   4         1        -1
        1   0   0  -2         0         0
        2   0 -10   4         0        -1
        3  11   0  41         1         0
        4  17   1   4         1         1
    """
    df_copy = df.copy()
    for feat in features:
        df_copy.loc[:, f'{feat}_sign_%'] = np.sign(df_copy.loc[:, feat])
    if print_signed_pcts_col:
        for num_f in features:
            print(df_copy[f'{num_f}_sign_%'].value_counts(normalize=True).round(3)*100)
            print()

    return df_copy
##############################


def find_const_and_null_cols(
        df: pd.DataFrame,
        verbose: int = 1,
        ignore_cols: list = None
) -> pd.DataFrame:
    """
    Function to drop columns with all constant or all null values.

    Args:
        df: pd.DataFrame
            The dataframe of interest.
        verbose: int
            Flag to inspect the columns dropped.
        ignore_cols: list
            Columns to be ignored from consideration.

    Returns:
        list
            List of all columns with either constant values or all null values.

    Examples:
        >>> df_dict = {'col1': [1, 1, 1, 1, 1], 'col2': [np.nan, np.nan, np.nan, np.nan, np.nan], 'col3': [1, np.nan, 1, np.nan, 1]}
        >>> df = pd.DataFrame(df_dict)
        >>> print(find_const_and_null_cols(df))
        Constant columns: ['col1', 'col3']
        <BLANKLINE>
        All null columns: ['col2']
        <BLANKLINE>
        ['col1', 'col3', 'col2']
    """
    df_output = df.copy()
    cols_const = list(df_output.columns[df_output.nunique() == 1])
    cols_all_null = list(df_output.columns[df_output.nunique() == 0])

    if ignore_cols:
        cols_const = [col for col in cols_const if col not in ignore_cols]
        cols_all_null = [col for col in cols_all_null if col not in ignore_cols]

    if verbose:
        print(f"Constant columns: {cols_const}\n")
        print(f"All null columns: {cols_all_null}\n")

    return cols_const + cols_all_null

##############################


def synthetic_data_generate(
        distribution: str,
        start_param: float,
        end_param: float = None,
        seed: int = 23,
        no_samples: int = 2500,
        plot_data: bool = True,
        color: str = 'cyan',
        bins: int = 50,
        figsize=(10, 8)
) -> np.array:
    """
    Function to generate synthetic data for 'normal', 'uniform', 'exponential', 'lognormal',
    'chisquare' and 'beta' distributions.

    Args:
        distribution : str
            The distribution for which the synthetic data is to be created.
        start_param : float
            First parameter of the distribution. Conforms to np.random.'distribution'.
        end_param : float
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
        allowed distribution strings.

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

    if distribution in allowed_dists:
        np.random.seed(seed=seed)
        if end_param:
            evaluation_string = f'np.random.{distribution}({start_param}, {end_param}, {no_samples})'
        else:
            evaluation_string = f'np.random.{distribution}({start_param}, {no_samples})'

        print(f"Evaluating: {evaluation_string}")
        synthetic_data_dist = eval(evaluation_string)
        if plot_data:
            _, ax = plt.subplots(figsize=figsize)
            ax.hist(synthetic_data_dist, bins=bins, color=color)
            plt.title(f"Synthetic {distribution} distribution")
            plt.show()

        return synthetic_data_dist

    raise NameError(
        f"{distribution} doesn't match any of the allowed distributions {allowed_dists}."
        )

##############################


if __name__=='__main__':
    import doctest
    doctest.testmod()

##############################
