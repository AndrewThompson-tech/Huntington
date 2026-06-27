# raw data folder
from data.raw_data.raw_etfs import get_etfs
from data.raw_data.raw_macros import get_macros

# normalize data folder
from data.norm_data.norm_etfs import normalize_etfs
from data.norm_data.norm_macros import normalize_macros

# get raw data
raw_etfs = get_etfs()
raw_macros = get_macros()

print(raw_etfs)
print('---------------------------------------')
print(raw_macros)
print('---------------------------------------')

# normalize data
norm_etfs = normalize_etfs(raw_etfs)
norm_macros = normalize_macros(raw_macros)

# print(norm_fred)
# print(norm_fred['Real_10Y_Yield'].to_string())
# print(norm_fred['Trade_Weighted_USD_Index'].to_string())


# norm_fred['Real_10Y_Yield'].to_json("macro_data.json")
# norm_fred['Trade_Weighted_USD_Index'].to_json("macro_data.json")
print(norm_etfs)
print('---------------------------------------')
print(norm_macros)
print('---------------------------------------')

# for col in norm_fred:
#     print(norm_fred[col].head(10))
#     print(norm_fred[col].tail(10))
