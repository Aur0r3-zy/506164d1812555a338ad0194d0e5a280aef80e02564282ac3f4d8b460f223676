import csv
import pandas as pd
import numpy as np


''' Record CSV heads, features of those traffic '''

def record_csv():
    with open((data_path+'02-14-2018.csv'), 'r') as f:
        reader = csv.reader(f)
        print(type(reader))
        result = list(reader)
        print(result[0])

        df = pd.DataFrame(result[0], columns=['Column Name'])
        df.to_excel(data_path+'heads.xlsx')
        print('----------Save DONE-----------')
    f.close()


# Change your data path here
data_path = 'E:\\Desktop\\Intrusion-Detection-SystemLearning\\CNN-Distillation-model\\data\\'
file_name = 'total.csv'


''' Write Data '''

def write_csv(path):
    print('Loading csv file: '+path)
    data = pd.read_csv(path, header=None, low_memory=False)
    return data


''' Get Data '''

def get_data():
    f1 = data_path+'01-03-2018.csv'
    fr1= write_csv(f1).drop([0])
    f2 = data_path+'02-03-2018.csv'
    fr2= write_csv(f2).drop([0])
    f3 = data_path+'14-02-2018.csv'
    fr3= write_csv(f3).drop([0])


    data_frame = [fr1, fr2, fr3]

    return data_frame


''' Merge data '''

def merge_data(data_frame):
    data = pd.concat(data_frame, ignore_index=True)

    # clear Nan and Infinity
    data = data[~data.isin([np.nan, np.inf, -np.inf]).any(axis=1)].dropna()

    return data


''' Merge data into total.csv '''

def merge_csv():
    data_frame = get_data()
    data = merge_data(data_frame)
    file = data_path+file_name
    data.to_csv(file, index=False, header=False)
    print('----------Total CSV Save Done---------')

''' Count Labels '''

def count_labels():
    raw_data = pd.read_csv(data_path+file_name, header=None, low_memory=False)
    index_num = raw_data.shape[1]-1
    # There are "Headers" in data, delete these data
    raw_data = raw_data.drop(raw_data[raw_data[index_num] == 'Label'].index, axis=0)
    print(raw_data[index_num].value_counts())

''' Main Program '''

merge_csv()
count_labels()
