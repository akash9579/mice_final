from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
import logging
import flask_monitoringdashboard as dashboard
from train_code.input import input
from train_code.pre_processing import pre_processing
#from validation import validation
from train_code.model import model
from validation.validation import validation


#set up the log file and logger
logger = logging.getLogger("train_path")
f_handler = logging.FileHandler('log_records/file.log')
f_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')#set the format
f_handler.setFormatter(formatter)
logger.addHandler(f_handler)

# setting app
app=Flask(__name__)
dashboard.bind(app)

# loading all classifier 
pickle_in = open("modules/DecisionTreeClassifier.pkl","rb")
classifier=pickle.load(pickle_in)
pickle_in1 = open("modules/KNeighborsClassifier.pkl","rb")
classifier1=pickle.load(pickle_in1)
pickle_in2 = open("modules/RandomForestClassifier.pkl","rb")
classifier2=pickle.load(pickle_in2)


@app.route('/')
def home():
    logger.warning("welcome to home page")
    return render_template('home.html')
 

@app.route('/', methods=['POST'])
def upload_file():   
    try:
        logger.warning("starting of loading the file")
        # here we load file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            path = "data/test/"+uploaded_file.filename
            uploaded_file.save(path) # here we can try to save data in upload folder
            logger.warning("user file store in test folder" )
            
            # here we are cheching if file having 81 columns are not
            logger.warning("starting validating user file" )
            validation_check = validation()
            column_names, noofcolumns=validation_check.valuesFromSchema()
            valid= validation_check.validateColumnLength(path,noofcolumns)
            # if valid is 0 then file not having 81 column
            # if valid is 1 then user uploaded correct file
            if valid == 0:
                # if file not having 81 column then show error message
                return render_template('error.html')
            else:
                return redirect(url_for('home')) 
            logger.warning("user uploaded file validatiom completed completed" )
        else:
                return redirect(url_for('home')) 
    except ValueError:
        logger.error("Error Occurred! %s" %ValueError)
    except KeyError:
        logger.error("Error Occurred! %s" %KeyError)
    except Exception as e:
        logger.error("Error Occurred! %s" %e)
 
        
@app.route('/train',methods=['GET','POST'])
def train():                                        
    try:
        logger.warning("starting of trainning ")
        logger.warning("data loading completed ")
        
        # loading the data into dataframe        
        a1=input()
        a1.input1()
        logger.warning("data loading completed" )
        
        #started the preprocessing of data
        b=pre_processing()
        b.data=a1.data#1080*82
        b.null_list=a1.null_list###49 
        b.preprocessing()
        logger.warning("data pre-processing completed" )
        
        # model methode used to create classifire and stored in module folder
        c=model()
        a2=input()
        a2.input1()
        c.data=b.data#its preprocess data1080*20
        c.data_original=a2.data#1080*82
        c.model()
        logger.warning("all 3 models succesfully trained")
        
        return redirect(url_for('home')) # going to home page only 
    except ValueError:
        logger.error("Error Occurred! %s" %ValueError)
    except KeyError:
        logger.error("Error Occurred! %s" %KeyError)
    except Exception as e:
        logger.error("Error Occurred! %s" %e)

@app.route('/predict',methods=['GET','POST'])
def predict():
    try:
        logger.warning("starting of prediction path using KNeighborsClassifier ")
        # first load user uploaded data into dataframe
        logger.warning("starting of loading user uploaded data ")
        df1=pd.read_csv('data/test/filecheck.csv') 
        data1=df1.drop(df1.columns[0], axis = 1)
        # store mouse_id in id 
        id=data1.MouseID
        id1=id.to_numpy()
        #calling input function for getting null_list
        a3=input()
        a3.input1()
        logger.warning("starting of pre-processing of user uploaded  data ")
        #start pre-processing user uploaded data
        b=pre_processing()
        b.null_list=a3.null_list
        b.input_preprocessing(data1)
        #final_data is the preprocesses data
        logger.warning("we got pre=processed data")
        final_data=b.data_pre
        my_prediction=classifier.predict(final_data)
        # following code is used for passing dict to html page
        dict1=[]
        for i in range (0,len(id1)-1):
            dict1.append(dict(id=id1[i],class1=my_prediction[i]))
            logger.warning(" prediction done using DecisionTreeClassifier ")
        return render_template('resultDecisionTreeClassifier.html',prediction = dict1) 
    except ValueError:
        logger.error("Error Occurred! %s" %ValueError)
    except KeyError:
        logger.error("Error Occurred! %s" %KeyError)
    except Exception as e:
        logger.error("Error Occurred! %s" %e)
        
@app.route('/predict1',methods=['GET','POST'])
def predict1():
    try:
        logger.warning("starting of prediction path using KNeighborsClassifier ")
        # first load user uploaded data into dataframe
        logger.warning("starting of loading user uploaded data ")
        df1=pd.read_csv('data/test/filecheck.csv') 
        data1=df1.drop(df1.columns[0], axis = 1)
        # store mouse_id in id 
        id=data1.MouseID
        id1=id.to_numpy()
        #calling input function for getting null_list
        a3=input()
        a3.input1()
        logger.warning("starting of pre-processing of user uploaded  data ")
        #start pre-processing user uploaded data
        b=pre_processing()
        b.null_list=a3.null_list
        b.input_preprocessing(data1)
        #final_data is the preprocesses data
        logger.warning("we got pre=processed data")
        final_data=b.data_pre
        my_prediction=classifier1.predict(final_data)
        
        # so here tr-training approch come into picture
        # here we creating newly predicted data and storing in new.csv which is used for retraing approch
        new=my_prediction.tolist()
        df=pd.read_csv('data/test/filecheck.csv') 
        data=df.drop(df1.columns[0], axis = 1)
        print(data1.shape)
        print(data1.shape)
        print(type(data1))
        #print(data1.shape)
        print(type(my_prediction))
        print(type(new))
        data['class'] = new
        new_data=data
        print(new_data.head(5))
        new_data.to_csv('data/train/new.csv')
        logger.warning("newly created data store in new.csv file")
        
        # following code is used for passing dict to html page
        dict2=[]
        for i in range (0,len(id1)-1):
            dict2.append(dict(id=id1[i],class1=my_prediction[i]))
            logger.warning(" prediction done using KNeighborsClassifier ")
        return render_template('resultKNeighborsClassifier.html',prediction = dict2) 
    except ValueError:
        logger.error("Error Occurred! %s" %ValueError)
    except KeyError:
        logger.error("Error Occurred! %s" %KeyError)
    except Exception as e:
        logger.error("Error Occurred! %s" %e)
        
@app.route('/predict2',methods=['GET','POST'])

def predict2():
    try:
        logger.warning("starting of prediction path using RandomForestClassifier ")
        # first load user uploaded data into dataframe
        logger.warning("starting of loading user uploaded data ")
        df1=pd.read_csv('data/test/filecheck.csv') 
        data1=df1.drop(df1.columns[0], axis = 1)
        # store mouse_id in id 
        id=data1.MouseID
        id1=id.to_numpy()
        #calling input function for getting null_list
        a3=input()
        a3.input1()
        logger.warning("starting of pre-processing of user uploaded  data ")
        #start pre-processing user uploaded data
        b=pre_processing()
        b.null_list=a3.null_list
        b.input_preprocessing(data1)
        #final_data is the preprocesses data
        logger.warning("we got pre=processed data")
        final_data=b.data_pre
        my_prediction=classifier2.predict(final_data)
        # following code is used for passing dict to html page
        dict3=[]
        for i in range (0,len(id1)-1):
            dict3.append(dict(id=id1[i],class1=my_prediction[i]))
            logger.warning(" prediction done using RandomForestClassifier ")
        return render_template('resultRandomForestClassifier.html',prediction = dict3) 
        
    except ValueError:
        logger.error("Error Occurred! %s" %ValueError)
    except KeyError:
        logger.error("Error Occurred! %s" %KeyError)
    except Exception as e:
        logger.error("Error Occurred! %s" %e)


@app.route('/about')
def about():
    logger.warning(" reading about problem statement ")
    return render_template('about.html')    

@app.route('/problem')
def problem():
    logger.warning(" reading about problem statement ")
    return render_template('problem.html')  

@app.route('/dashboard')
def dashboard():
    logger.warning(" reading about problem statement ")
    return redirect(url_for('dashboard')) # going to home page only 


@app.errorhandler(404)
def error404(error):
    return render_template('error.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000,DEBUG=True)
    
 
