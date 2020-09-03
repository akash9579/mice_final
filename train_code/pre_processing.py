from sklearn import preprocessing 
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd



#set up the log file and logger
logger = logging.getLogger("train_path")
f_handler = logging.FileHandler('log_records/file_pre.log')
f_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')#set the format
f_handler.setFormatter(formatter)
logger.addHandler(f_handler)


class pre_processing:

    null_list=[]
    
    def preprocessing(self):
        """
           Class Name: pre_processing
           Method Name: preprocessing
           Description: This method used for preprocessing the original data which covert 1080*82 data into 1080*20.
                        in this methd we are performing following  
                        1. handling null value
                        2. converting catogorial data into integral data
                        3. handling outlire
                        4. perform PCA 
           Output: we get clean data stored in data_pre dataframe
           On Failure: Raise Exception
        """
        try:
  
            logger.warning("starting of pre-processing function")
            data=self.data
            null_list=self.null_list

            logger.warning("1st we perform handling of null data")
            for i in range (0,len(null_list)):
                data[null_list[i]]=data[null_list[i]].fillna(data[null_list[i]].mean())      
                
            logger.warning("2nd we convert catogorial data into integer")
            label_encoder = preprocessing.LabelEncoder() 
            data.drop(['MouseID'],axis=1,inplace=True)
            data.drop(['class'],axis=1,inplace=True)
            data['Genotype']= label_encoder.fit_transform(data['Genotype']) 
            data['Treatment']= label_encoder.fit_transform(data['Treatment']) 
            data['Behavior']= label_encoder.fit_transform(data['Behavior'])   

            logger.warning("3rd we handle outlire ")
            for i in range(1,len(data.columns)-5):
                IQR=data[data.columns[i]].quantile(0.75)-data[data.columns[i]].quantile(0.25)
                upper_bridge=data[data.columns[i]].quantile(0.75)+(IQR*1.5)
                data.loc[data[data.columns[i]]>=upper_bridge,data.columns[i]]=upper_bridge
                logger.warning("data preprocessing completed")
                
            logger.warning("4th we perfrom PCA ")
            scaler=StandardScaler()
            scaler.fit(data)
            scaled_data=scaler.transform(data)
            pca=PCA(n_components=20)
            pca.fit(scaled_data)
            data_p=pca.transform(scaled_data)
            data_pca=pd.DataFrame(data_p)
            self.data=data_pca
            
        except ValueError:
            logger.error("Error Occurred! %s" %ValueError)
        except KeyError:
            logger.error("Error Occurred! %s" %KeyError)
        except Exception as e:
            logger.error("Error Occurred! %s" %e)

        
            
    def input_preprocessing(self,input_data):
        """
           Class Name: pre_processing
           Method Name: input_preprocessing
           Description: This method used for preprocessing the user data.
                        in this methd we are performing following  
                        1. handling null value
                        2. converting catogorial data into integral data
                        3. handling outlire
                        4. perform PCA 
           Output: we get clean data stored in data_pre dataframe
           On Failure: Raise Exception
        """
        try:
  
            logger.warning("starting of pre-processing function")
            null_list=self.null_list
            data=input_data

            logger.warning("1st we perform handling of null data")
            for i in range (0,len(null_list)):
                data[null_list[i]]=data[null_list[i]].fillna(data[null_list[i]].mean())            

            logger.warning("2nd we convert catogorial data into integer")
            label_encoder = preprocessing.LabelEncoder() 
            data.drop(['MouseID'],axis=1,inplace=True)
            data['Genotype']= label_encoder.fit_transform(data['Genotype']) 
            data['Treatment']= label_encoder.fit_transform(data['Treatment']) 
            data['Behavior']= label_encoder.fit_transform(data['Behavior'])   
 
            logger.warning("3rd we handle outlire ")
            for i in range(1,len(data.columns)-5):
                IQR=data[data.columns[i]].quantile(0.75)-data[data.columns[i]].quantile(0.25)
                upper_bridge=data[data.columns[i]].quantile(0.75)+(IQR*1.5)
                data.loc[data[data.columns[i]]>=upper_bridge,data.columns[i]]=upper_bridge
                logger.warning("data preprocessing completed")
                
            logger.warning("4th we perfrom PCA ")
            scaler1=StandardScaler()
            scaler1.fit(data)
            scaled_data=scaler1.transform(data)
            pca=PCA(n_components=20)
            pca.fit(scaled_data)
            data_p=pca.transform(scaled_data)
            data_pca=pd.DataFrame(data_p)
            self.data_pre=data_pca
 
        except ValueError:
            logger.error("Error Occurred! %s" %ValueError)
        except KeyError:
            logger.error("Error Occurred! %s" %KeyError)
        except Exception as e:
            logger.error("Error Occurred! %s" %e)



