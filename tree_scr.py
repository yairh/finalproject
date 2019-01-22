from finalproject.class_dataset import ChestDataset
import pandas as pd

data_dir = str(input('input directory:'))
csvfile = str(input('input csv path:'))
df = pd.read_csv(csvfile)
df_uni = df[~df['Finding Labels'].str.contains('\|')]  
dataset = ChestDataset(data_dir,df)
try:
    dataset.reset_folder()
    print('Folder reset')
except Exception as e:
    print(e)

