"""Useful functions to prepare features for modeling"""

import pandas as pd
import numpy as np

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)


def read_features(path_to_features, features_list):
    
    assert len(features_list) > 0
    
    result = pd.read_parquet(path_to_features + features_list[0])
    id_cols = ['is_train', 'PROVIDER_NPI']
    
    for id_col in id_cols:
        if id_col not in result.columns:
            print(f'{id_col} not in {features_list[0]}, exiting.')
            return
    
    if len(features_list) > 1:
        for fts_file in features_list[1:]:
            
            new_df = pd.read_parquet(path_to_features + fts_file)
            
            if new_df.PROVIDER_NPI.dtypes == float:
                new_df['PROVIDER_NPI'] = new_df.PROVIDER_NPI.astype(int).astype(str)
            
            for id_col in id_cols:
                if id_col not in new_df.columns:
                    print(f'{id_col} not in {ft_file}, exiting.')
                    return
            
            dupl_columns = [i for i in new_df.columns if i in result.columns and i not in id_cols]
            if len(dupl_columns) > 0:
                
                print(f'\tFound duplicating columns in {fts_file}:', dupl_columns)
                print('\tdeleting them.')
                new_df.drop(dupl_columns, axis=1, inplace=True)
                
            result = result.merge(new_df, how='outer')
            
    return result


def calculate_psi_for_features(features_df):
    
    features_to_test = [c for c in features_df.columns if c not in ['is_train', 'PROVIDER_NPI']]
    
    psi_tuples = []
    
    for feature in features_to_test:
        na_share_expected = features_df.loc[features_df.is_train == 1, feature].isnull().mean()
        na_share_actual = features_df.loc[features_df.is_train == 0, feature].isnull().mean()
        expected = features_df.loc[features_df.is_train == 1, feature].dropna().values
        actual = features_df.loc[features_df.is_train == 0, feature].dropna().values
        psi = calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0)
        psi_tuples.append((feature, psi, na_share_expected, na_share_actual))
    
    result = pd.DataFrame(psi_tuples, columns = ['feature','PSI', 'NA_share_train', 'NA_share_actual'])
    result['NA_shares_diff'] = np.abs(result.NA_share_train - result.NA_share_actual)
    return result.sort_values('PSI', ascending=False)


def get_too_high_nan_shares(features_df, nan_share_threshold = .95):
    
    nan_shares = features_df.drop(['is_train', 'PROVIDER_NPI'], axis=1).isnull().mean().reset_index()
    return nan_shares.loc[nan_shares[0] > nan_share_threshold, 'index'].values.tolist()

def get_high_correlated_features(features_df, correlation_threshold = .975):
    
    corr_matrix = features_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    return to_drop

def prepare_features(path_to_features, features_list, path_to_target):
    
    features_df = read_features(path_to_features, features_list)
    print('Loaded data shape: ', features_df.shape)
    
    psi_summary = calculate_psi_for_features(features_df)
    too_high_psi = psi_summary.loc[psi_summary.PSI > .2, 'feature'].values.tolist()
    
    if len(too_high_psi) > 0:
        print(f'{len(too_high_psi)} features deleted due to PSI > .2:')
        for thp in too_high_psi:
            psi_val = psi_summary.loc[psi_summary.feature == thp, 'PSI'].max()
            print(f'\t {thp}: {psi_val}')
            
    features_df.drop(too_high_psi, axis=1, inplace=True)
    
    to_high_na_diff = psi_summary.loc[psi_summary.NA_shares_diff > .2, 'feature'].values.tolist()
    
    if len(to_high_na_diff) > 0:
        print(f'{len(to_high_na_diff)} features deleted due to train/test NA share difference > .2:')
        for thp in to_high_na_diff:
            na_diff_val = psi_summary.loc[psi_summary.feature == thp, 'NA_shares_diff'].max()
            print(f'\t{thp}: {na_diff_val}')
        
        features_df.drop(to_high_na_diff, axis=1, inplace=True)
    
    too_high_nan_shares = get_too_high_nan_shares(features_df)
    if len(too_high_nan_shares) > 0:
        print(f'{len(too_high_nan_shares)} features deleted due to NAN share > 95%')
        features_df.drop(too_high_nan_shares, axis=1, inplace=True)
    
    highly_correlated = get_high_correlated_features(features_df.drop(['is_train', 'PROVIDER_NPI'], axis=1))
    if len(highly_correlated) > 0:
        print(f'{len(highly_correlated)} features deleted due to high correlation with other features')
        features_df.drop(highly_correlated, axis=1, inplace=True)
        
    for column in features_df.columns:
    
        if 'share' in column or 'avg' in column or 'median' in column or 'std' in column or 'count' in column or 'nunique' in column:
            features_df[column].fillna(0, inplace=True)
    
    print('Remaining data shape: ', features_df.shape)
    print('Columns with missing values: ', features_df.isnull().max().sum())
    
    
    train_df = features_df[features_df.is_train == 1].copy()
    target = pd.read_parquet(path_to_target + 'train_target.parquet').drop('SPECIALTY2', axis=1)
    train_df = train_df.merge(target, how='inner').drop('is_train', axis=1)
    
    test_df = features_df[features_df.is_train == 0].copy()
    target = pd.read_parquet(path_to_target + 'test_target.parquet').drop('SPECIALTY2', axis=1)
    
    missing_in_features = set(target.PROVIDER_NPI.unique()) - set(test_df.PROVIDER_NPI.unique())
    if len(missing_in_features) > 0:
        print(f'ALERT! {len(missing_in_features)} NPI missing in test features!')
    test_df = test_df.merge(target, how='inner').drop('is_train', axis=1)
    
    return train_df, test_df
