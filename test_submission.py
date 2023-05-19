import pandas as pd

def test_submission_file(df_correct: pd.DataFrame, df_test: pd.DataFrame):
    """test function to check format of the submission file"""
    
    # Two columns: PROVIDER_NPI, PROBABILITY (capital letter)
    if set(df_test.columns) != {'PROBABILITY', 'PROVIDER_NPI'}:
        print("Wrong columns")
    
    # HCP ID set matches template submission file
    elif set(df_test['PROVIDER_NPI']) != set(df_correct['PROVIDER_NPI']):
        print("Wrong HCP ID list")
    
    # No duplicated HCP
    elif df_test.shape[0] != df_correct.shape[0]:
        print("Duplicated HCP ID found")
    
    # Probability is between [0,1]
    elif (df_test['PROBABILITY'].max() > 1) | (df_test['PROBABILITY'].min() < 0):
        print("Wrong probability range")
        
    else: print("PASS!")