import pandas as pd
import logging


#set up the log file and logger
logger = logging.getLogger("train_path")
f_handler = logging.FileHandler('log_records/file_input.log')
f_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')#set the format
f_handler.setFormatter(formatter)
logger.addHandler(f_handler)


class input:
        
    null_list=[]
    accuracy=[]
    
    def input1(self):
        """
           Class Name: Input
           Method Name: Input
           Description: This method used for loading original data and newly added data into dataframe.
           Output: None
           On Failure: Raise Exception
        """
        try:
            logger.warning("starting of input function ")
            data1 = pd.read_excel('data/train/Data_Cortex_Nuclear.xls')#original data
            df1=pd.read_csv('data/train/new.csv') #newly added data
            data2=df1.drop(df1.columns[0], axis = 1)
            result = pd.concat([data1,data2])
            data=result
            # data dataframe contain final data after merging original and newly added data
            cat_features=[i for i in data.columns if data.dtypes[i]=='object'] 
            for i in range (0,data.isnull().sum().shape[0]-1):
                if(data.isnull().sum()[i]>0):
                    self.null_list.append(data.isnull().sum().index[i])
                    self.data=data
                    self.cat_features=cat_features
        except ValueError:
            logger.error("Error Occurred! %s" %ValueError)
        except KeyError:
            logger.error("Error Occurred! %s" %KeyError)
        except Exception as e:
            logger.error("Error Occurred! %s" %e)


      

