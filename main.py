import pandas as pd

data = pd.read_csv("horse.csv")
data.head()

#1. SDV  (https://sdv.dev/SDV/index.html)
# demo data
import sdv.demo as demo
demos = demo.get_available_demos()
demos['name']
from sdv.demo import load_tabular_demo
metadata, train_data = load_tabular_demo('trains_v1', metadata=True)
train_data.head()
train_data.columns

# tabularPreset
from sdv.lite import TabularPreset
model = TabularPreset(name='FAST_ML', metadata=metadata)
model.fit(data)

synthetic_data = model.sample(63)
synthetic_data.head()

from sdv.evaluation import evaluate
evaluation_TP  = evaluate(synthetic_data, data, metrics=['CSTest', 'KSTest'], aggregate=False)

## GaussianCopula (https://github.com/sdv-dev/SDV/blob/master/tutorials/single_table_data/01_GaussianCopula_Model.ipynb)
from sdv.tabular import GaussianCopula
model = GaussianCopula(primary_key='id') # so it does not create duplicates
model.fit(data)

gaussian_data = model.sample(63)
gaussian_data.head()

from sdv.evaluation import evaluate
evaluation_GC  = evaluate(gaussian_data, data, metrics=['CSTest', 'KSTest'], aggregate=False)

def multiple_dfs(df_list, sheets, file_name, spaces):
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')
    row = 0
    for dataframe in df_list:
        dataframe.to_excel(writer,sheet_name=sheets,startrow=row , startcol=0)
        row = row + len(dataframe.index) + spaces + 1
    writer.save()

data_summary = data.describe()
tabular_summary = synthetic_data.describe()
gaussian_summary = gaussian_data.describe()
dfs = [data_summary, tabular_summary, gaussian_summary]
multiple_dfs(dfs, 'Validation', 'test1.xlsx', 1)



