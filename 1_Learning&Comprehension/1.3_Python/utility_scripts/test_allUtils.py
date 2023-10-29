import datetime
import unittest

import numpy as np
import pandas as pd

import utils
import utils_modelling

class TestUtility(unittest.TestCase):
    
    def setUp(self):
        self.list_ex = ['A', 1, 2, 9, 'D', '90', 'X', np.nan]
        self.tup_ex = (np.nan, 2, 2, 'X', 'Z', 'Z') 
        self.set_ex = {1, 2, '90', 'X', 'Y', 'Z'}
        self.set_ex_emp = {}
        
        df1 = pd.DataFrame(data=[[1, -1], [-1, -1]], columns=['a', 'b'])
        df2 = pd.DataFrame(data=[[1, 1], [1, -1]], columns=['c', 'd'])
        df3 = pd.DataFrame(data=[[-1, 1, 1], [0, 1, -1], [0, 0, 1]], columns=['d', 'b', 'c'])
        df4 = pd.DataFrame(data=[[-1, 1, 0], [0, -1, 1]], columns=['a', 'd', 'e'])
        self.df_dict = {
            'df1': df1,
            'df2': df2,
            'df3': df3,
            'df4': df4
        }
        
        self.dfcn = pd.DataFrame({
            'col1': [1, 1, 1, 1, 1],
            'col2': [np.nan, np.nan, np.nan, np.nan, np.nan],\
            'col3': [1, np.nan, 1, np.nan, 1],
            'col4': np.arange(5)
        })
    
    def test_get_common_elements(self):
        self.assertEqual(utils.get_common_elements(self.list_ex, self.tup_ex), {'X', 2, np.nan})
        self.assertEqual(utils.get_common_elements(self.list_ex, self.set_ex), {1, 2, '90', 'X'})
        self.assertEqual(bool(utils.get_common_elements(self.list_ex, self.set_ex_emp)), False)
        self.assertEqual(utils.get_common_elements(self.tup_ex, self.set_ex), {2, 'X', 'Z'})
        self.assertEqual(bool(utils.get_common_elements(self.tup_ex, self.set_ex_emp)), False)
        self.assertEqual(bool(utils.get_common_elements(self.set_ex, self.set_ex_emp)), False)
        
    def test_common_cols_by_name_dfs(self):
        output_t1 = {
            ('df1', 'df3'): {'b'},
            ('df1', 'df4'): {'a'},
            ('df2', 'df3'): {'c', 'd'},
            ('df2', 'df4'): {'d'},
            ('df3', 'df4'): {'d'}
        }
        output_t2 = {('df2', 'df3', 'df4'): {'d'}}
        output_t3 = {}
        
        self.assertEqual(utils.common_cols_by_name_bw_dfs(self.df_dict), output_t1)
        self.assertEqual(utils.common_cols_by_name_bw_dfs(self.df_dict, comb_size=3), output_t2)
        self.assertEqual(utils.common_cols_by_name_bw_dfs(self.df_dict, comb_size=4), output_t3)
        
        
    def test_find_const_and_null_cols_df(self):
        self.assertEqual(set(utils.find_const_and_null_cols_df(self.dfcn)), set(['col1', 'col3', 'col2']))
        self.assertEqual(set(utils.find_const_and_null_cols_df(self.dfcn, ignore_cols=['col2'])), set(['col3', 'col1']))
        
if __name__=='__main__':
    unittest.main()