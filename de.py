import numpy as np
import pandas as pd
from sklearn import preprocessing

def get_data():
# Read excel file
    File = pd.ExcelFile('data.xlsx')
    File.sheet_names
    df = File.parse('D1')
# seprate data and label
#    x = np.array(df.drop(['label'],1))
#    y = np.array(df['label'])
# preprocessing
    x = preprocessing.scale(df) # train Data
    x1 = x[0:300, 0:21] 
#    y_train = y[0:300]
    x2 = x[0:1500, 0:21] 
#    x2in3 = x1[2150:2450,:]
#    x2 = np.concatenate((x2, x2in3), axis=0)
#
#    x2 = x1[0:3000,:]
##       
    x_train = x1 
    x_test = x2
    return[x_train,x_test]