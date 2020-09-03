# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:45:47 2020

@author: CSE
"""


import json
import pandas as pd


class validation:
    schema_path="train_schema.json"

    def valuesFromSchema(self):
        with open(self.schema_path, 'r') as f:
            dic = json.load(f)
            f.close()
        column_names = dic['ColName']
        NumberofColumns = dic['NumberofColumns']



        return  column_names, NumberofColumns
    
    
    
    def validateColumnLength(self,path,NumberofColumns):
        df1=pd.read_csv(path)   #    here we giving bydeafult file that need to change
        data1=df1.drop(df1.columns[0], axis = 1) 
        if data1.shape[1] == NumberofColumns:
            valid = 1
        else:
            valid = 0
        return valid
            
