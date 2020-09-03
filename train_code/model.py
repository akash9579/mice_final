from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import logging


#set up the log file and logger
logger = logging.getLogger("train_path")
f_handler = logging.FileHandler('log_records/file_model.log')
f_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')#set the format
f_handler.setFormatter(formatter)
logger.addHandler(f_handler)


class model:
    
    def model(self):
        """
           Class Name: model
           Method Name: model
           Description: This function used for building all three models.
           Output: store all three modules in module folder by using pickle library
           On Failure: Raise Exception
        """
        
        try:
            logger.warning("starting of model creation function")
            data1=self.data
            data=self.data_original
            
            #getting X,Y data for giving to train-test-split function
            x=data1
            y=data.pop('class')
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
            
            #DecisionTreeClassifier
            logger.warning("model for DecisionTreeClassifier is created")
            dt = tree.DecisionTreeClassifier()
            model = dt.fit(X_train, y_train)
            prad = model.predict(X_test)
            final=accuracy_score(y_test, prad)
            print("DecisionTreeClassifier")
            print(final)        
            pickle_out = open("modules/DecisionTreeClassifier.pkl","wb")
            pickle.dump(model, pickle_out)
            pickle_out.close()
            
            #KNeighborsClassifier
            logger.warning("model for KNeighborsClassifier is created")
            model1 = KNeighborsClassifier(n_neighbors=3)
            model1.fit(X_train, y_train)
            prad1=model1.predict(X_test)
            final1=accuracy_score(y_test, prad1)
            print("KNeighborsClassifier")
            print(final1)  
            pickle_out = open("modules/KNeighborsClassifier.pkl","wb")
            pickle.dump(model1, pickle_out)
            pickle_out.close()
            
            
            #RandomForestClassifier
            logger.warning("model for RandomForestClassifier is created")
            model2 = RandomForestClassifier(max_depth=2, random_state=0)
            model2.fit(X_train, y_train)
            prad2=model2.predict(X_test)
            final2=accuracy_score(y_test, prad2)   
            print("RandomForestClassifier")
            print(final2)  
            pickle_out = open("modules/RandomForestClassifier.pkl","wb")
            pickle.dump(model2, pickle_out)
            pickle_out.close()
            

        

            logger.warning("all 3 model gets created and stored in modules folder succesfully")

        except ValueError:
            logger.error("Error Occurred! %s" %ValueError)
        except KeyError:
            logger.error("Error Occurred! %s" %KeyError)
        except Exception as e:
            logger.error("Error Occurred! %s" %e)



