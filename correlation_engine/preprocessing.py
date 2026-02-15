import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import sys
print(sys.path)
def enforce_stationary(master_df: pd.DataFrame, macro_columns: list, etf_columns: list): 
   master_df_transformed = master_df.copy() # just incase we want to use both df's in the future
   etf_transformations = {
      etf: [0, False] for etf in etf_columns # assume non-stationary (False)
   }
   macro_transformations = {
      macro: [0, False] for macro in macro_columns 
   }

   # ensure all etf columns are stationary
   for etf in etf_columns:      
      curr_etf = master_df[etf]
      
      if isStationary(curr_etf): # true == stationary
          etf_transformations[etf][1] = True
          continue
      else: 
         curr_etf = curr_etf.pct_change() # convert from etf monthly price levels to etf returns
         etf_transformations[etf][0] += 1

         if isStationary(curr_etf):
            etf_transformations[etf][1] = True
            master_df_transformed[etf] = curr_etf
            continue
         else:
            curr_etf = curr_etf.diff() 
            etf_transformations[etf][0] += 1
           
            if isStationary(curr_etf):
               etf_transformations[etf][1] = True  
            
            master_df_transformed[etf] = curr_etf
   
   # ensure all macro columns are stationary
   for macro in macro_columns:      
      curr_macro = master_df[macro]
      
      if isStationary(curr_macro): 
         macro_transformations[macro][1] = True
         continue
      else: 
         curr_macro = curr_macro.diff() 
         macro_transformations[macro][0] += 1

         if isStationary(curr_macro):
            macro_transformations[macro][1] = True
            master_df_transformed[macro] = curr_macro
            continue
         else:
           curr_macro = curr_macro.diff() 
           macro_transformations[macro][0] += 1
           
           if isStationary(curr_macro):
               macro_transformations[macro][1] = True

           master_df_transformed[macro] = curr_macro
   
   return master_df_transformed, macro_transformations, etf_transformations 

# Helper function
def isStationary(series: pd.Series) -> bool:
   adf_test_results = adfuller(series.dropna()) # Returns a list of info about that pandas series
   p_value = adf_test_results[1] 
   significance_level = 0.05 # 95th percentile
   
   return p_value < significance_level # If true then data is stationary
