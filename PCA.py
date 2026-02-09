import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def dynamic_pca(df, correlation_threshold=0.8, variance_explained=0.9):
    """
    Perform PCA on correlated groups of macro variables and retain uncorrelated variables.
    
    df- master macro sheet
    correlation_thresehold- minimum amount of correlation needed before PCA is performed
    variance_explained- PCA variable and really complicated. Basically how much information is retained between the
        two highly correlated variables
        
    returns 
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Step 1: Identify clusters of correlated variables
    correlated_groups = []
    used_cols = set()
    
    for col in corr_matrix.columns:
        if col in used_cols:
            continue
        group = set(corr_matrix.index[corr_matrix[col] > correlation_threshold])
        if len(group) > 1:
            correlated_groups.append(sorted(group))  # sort for consistent naming
            used_cols.update(group)

    print(f"Correlated Groups: {correlated_groups}")
    
    # Step 2: Apply PCA to each correlated group
    pca_results = pd.DataFrame(index=df.index)
    
    for group in correlated_groups:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[group])
        
        pca = PCA()
        pca.fit(scaled_data)
        
        # Determine number of components to keep
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.searchsorted(cum_var, variance_explained) + 1
        
        # Transform and add components to results
        transformed = pca.transform(scaled_data)[:, :n_components]
        for i in range(n_components):
            # Always use sorted column names + PC number
            col_name = f"{'_'.join(group)}_PC{i+1}"
            
            # Ensure uniqueness (in case of rerun)
            suffix = 1
            orig_col_name = col_name
            while col_name in pca_results.columns:
                col_name = f"{orig_col_name}_{suffix}"
                suffix += 1
                
            pca_results[col_name] = transformed[:, i]
    
    # Step 3: Add uncorrelated variables
    uncorrelated_cols = [c for c in numeric_cols if c not in used_cols]
    pca_results = pd.concat([pca_results, df[uncorrelated_cols]], axis=1)
    
    return pca_results

if __name__ == "__main__":
    MACRO = pd.read_csv("all_macros.csv", index_col="observation_date", parse_dates=True)
    pca_macro = dynamic_pca(MACRO, correlation_threshold=0.8, variance_explained=0.95)
    print(pca_macro.head())
    pca_macro.to_csv('pca_macros.csv')

