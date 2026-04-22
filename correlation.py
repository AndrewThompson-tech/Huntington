import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from data_cleanse import * 
import os


def correlation(pd, ticker):
    '''
    Finds correlation between specific ETF and macro data
    '''
    corr_matrix = pd.select_dtypes(include='number').corr()

    print(corr_matrix)

    plt.figure(figsize=(8,8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Greens')
    plt.title(f'Correlation Matrix of Macroeconomic Variables and {ticker}')
    plt.savefig(f'plots/{ticker}.png')
    plt.show()


def graph(MACRO, ETF,  ETF_name, MACRO_name):
    '''
        MACRO- the macro df, typically from master_macro_table.csv
        ETF- ETF df
        ETF_name- string you want displayed, ticker will do
        MACRO_name- string you want displayed for macro measurement

        problems: units, not every macro is the same
    '''
    # Put into one table
    data = pd.concat([ETF, MACRO], axis=1)
    data.columns = [f'{ETF_name}', f'{MACRO_name}']

    # visualize
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(data.index, data[f'{ETF_name}'], color='tab:blue', label=f'{ETF_name}')
    ax1.set_ylabel(f'{ETF_name} Price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(data.index, data[f'{MACRO_name}'], color='tab:red', label=f'{MACRO_name}')
    ax2.set_ylabel(f'{MACRO_name} Price', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(f'Quarterly Closing Prices: {ETF_name} vs {MACRO_name}')
    fig.tight_layout()
    plt.savefig(f'plots/{ETF_name}_vs_{MACRO_name}.png')
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
from data_cleanse import *
from correlation_engine.engine import run_correlation_engine
from correlation_engine.correlation import correlation
from PCA import dynamic_pca


def graph_dynamic(PROCESSING, TABLE_CONFIG, etf,
                  use_lag=False,
                  use_pca=False,
                  corr_threshold=0.80,
                  variance_explained=0.90,
                  stability_threshold=0.50,
                  normalize=True,
                  max_cols=6):
    """
    Graph ETFs vs macros using your existing pipeline.

    This function mirrors your modeling workflow so visuals match your models.
    """

    # --- Load data (same as your models) ---
    MACRO = master_table(TABLE_CONFIG, PROCESSING, "all_macros")
    ETF = fix_pd(etf)

    # Use returns (consistent with linear model)
    ETF["Close"] = ETF["Close"].pct_change()

    # Merge
    m_table = MACRO.merge(ETF[['Close']], on='observation_date', how='left')
    m_table = m_table[:240]

    y = m_table["Close"]

    macros_for_corr = list(MACRO.columns)
    yearly_period, lags = 5, 12

    # --- Apply lag logic if enabled ---
    if use_lag:
        run_correlation_engine(
            m_table,
            macros_for_corr,
            ["Close"],
            yearly_period,
            lags,
            generate_config=True
        )

        m_table, valid_lag = apply_lag(
            "optimal_lags.json",
            m_table,
            stability_threshold=stability_threshold
        )

        print("Valid lags:", valid_lag)

    # Separate X and y
    X = m_table.drop(columns=["Close"])

    # --- Apply PCA if enabled ---
    if use_pca:
        X = dynamic_pca(
            X,
            correlation_threshold=corr_threshold,
            variance_explained=variance_explained
        )
        print("Using PCA features:", X.columns.tolist())

    # --- Limit number of plotted columns (avoid clutter) ---
    if len(X.columns) > max_cols:
        print(f"Too many features ({len(X.columns)}), truncating to first {max_cols}")
        X = X.iloc[:, :max_cols]

    # Combine back for plotting
    plot_df = pd.concat([y, X], axis=1)

    # --- Normalize if needed (CRITICAL for macros vs returns) ---
    if normalize:
        plot_df = plot_df.dropna()
        plot_df = plot_df / plot_df.iloc[0] * 100

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # ETF (target)
    ax.plot(plot_df.index, plot_df["Close"], label="ETF (Close)", linewidth=2)
    ax.set_ylabel("ETF")

    # Macros / features
    for col in plot_df.columns:
        if col != "Close":
            ax.plot(plot_df.index, plot_df[col], linestyle="dashed", alpha=0.7, label=col)

    plt.title(f"ETF vs Macro Features ({'PCA' if use_pca else 'Raw'})")
    plt.legend(loc="best")
    plt.tight_layout()

    save_name = f"plots/workflow_plot_{'lag' if use_lag else 'nolag'}_{'pca' if use_pca else 'nopca'}.png"
    plt.savefig(save_name)

    plt.show()

def build_etf_macro_correlation(etf_dir="data/raw_data/ETFs", macro_dir="data/raw_data", save_path="plots/correlation_matrix.png"):
    """
    Builds a correlation matrix of all ETFs and macroeconomic data.
    """

    # ---- Load all macro files ----
    macro_files = [f for f in os.listdir(macro_dir) if f.endswith(".csv") and "ETFs" not in f]
    macro_dfs = []
    for file in macro_files:
        df = fix_pd(os.path.join(macro_dir, file))
        df = df.select_dtypes(include="number")
        
        # Fix naming: only append file name if multiple columns exist
        if len(df.columns) > 1:
            df.columns = [f"{file.replace('.csv','')}_{col}" for col in df.columns]
        else:
            df.columns = [df.columns[0]]  # keep original column name
        
        macro_dfs.append(df)
    
    macro_df = pd.concat(macro_dfs, axis=1)

    # ---- Load all ETF files ----
    etf_files = [f for f in os.listdir(etf_dir) if f.endswith(".csv")]
    etf_dfs = []
    for file in etf_files:
        df = fix_pd(os.path.join(etf_dir, file))
        if "Close" in df.columns:
            df = df[["Close"]]
        df.columns = [file.replace(".csv", "")]
        etf_dfs.append(df)

    etf_df = pd.concat(etf_dfs, axis=1)

    # ---- Merge ETFs + Macro ----
    combined_df = pd.concat([etf_df, macro_df], axis=1)
    combined_df = combined_df.dropna(how="any")

    # ---- Correlation ----
    corr_matrix = combined_df.corr()

    # ---- Plot ----
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=False, cmap="RdYlGn", center=0)
    plt.title("Correlation Matrix: ETFs vs Macros")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return corr_matrix

if __name__ == "__main__":
    corr_matrix = build_etf_macro_correlation()