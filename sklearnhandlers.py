#!/usr/bin/python

from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

from sklearn.neighbors import KNeighborsClassifier
import pickle
from bson.binary import Binary
import json
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.layers import Dense, Activation, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adagrad,Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):

        # Authentication
        # method is defined in basehandler
        # it's very simple authentication method. I get a key called "pass" from the cookies, and make sure that it matches the very specific key defined. 
        if False == self.check_cookie():
            self.write_json({"error": "authentication failed"})
            return

        '''Save data point and class label to database
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        label = data['label']
        sess  = data['dsid']

        # if it's a non-empty string, it can write into db
        if vals :
            dbid = self.db.labeledinstances.insert(
                {"feature":vals,"label":label,"dsid":sess}
                );
        
        self.write_json({"id":str(dbid),
            "feature":[str(len(vals))+" Points Received",
                    "min of: " +str(min(vals)),
                    "max of: " +str(max(vals))],
            "label":label})

class RequestNewDatasetId(BaseHandler):
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        # Authentication 
        if False == self.check_cookie():
            self.write_json({"error": "authentication failed"})
            return

        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a == None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

class UpdateModelForDatasetId(BaseHandler):
    
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        # Authentication 
        if False == self.check_cookie():
            self.write_json({"error": "authentication failed"})
            return

        # modle id
        dsid = self.get_int_arg("dsid",default=0)
        # it determines how many epochs the model shoudl run
        epochs = self.get_int_arg("epochs",default=1)
        # create feature vectors from database
        f=[];
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            f.append(a['feature'])

        # create label vector from database
        l=[];
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            l.append(int(a['label']))

        # fit the model to the data
        # here I'll use two RNN models, which are LSTM and GRU both with conceptNet embedding where I learned from Machine Learning in Python class.
        # To save a ton of codes here, I load the model I have trained from another jupyter-notebook(https://github.com/amor-tsai/MachineLearningNotebooks/blob/master/Lab7-ext.ipynb)
 
        if dsid == 1 :
            # use pre-trained LSTM model with embedding layer
            if (dsid not in self.clf.keys()) :
                self.clf[dsid] = load_model("lstm_conceptNet.h5")
            
        else :
            # use pre-trained GRU model with embedding layer
            if (dsid not in self.clf.keys()) :
                self.clf[dsid] = load_model("gru_conceptNet.h5")
            
        
        model = self.clf[dsid]
        acc = -1;
        if l and model:
            # loading tokenizer
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            
            # vectorization
            sequences = tokenizer.texts_to_sequences(f)

            # padding sequence
            data_for_train = pad_sequences(sequences,maxlen = 35)
            
            l = np.array(l)

            model.fit(
                data_for_train,
                l,
                epochs = epochs,
                verbose = 0,
                batch_size = 10,
            ) # training
            lstar = np.around(model.predict(data_for_train).flatten())
            acc = sum(lstar==l)/float(len(l))

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy":acc})

class PredictOneFromDatasetId(BaseHandler):
    def post(self):

        # Authentication 
        if False == self.check_cookie():
            self.write_json({"error": "authentication failed"})
            return

        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))    

        vals = data['feature'];
        dsid  = data['dsid']

        # load the model from the file(load_model)
        # To save a ton of codes here, I just use the model I trained from another jupyter-notebook. 
        # https://github.com/amor-tsai/MachineLearningNotebooks/blob/master/Lab7-ext.ipynb
        if(dsid not in self.clf.keys()) :
            print('Loading Model From File')
            if dsid == 1 :
                # use pre-trained LSTM model with embedding layer
                self.clf[dsid] = load_model("lstm_conceptNet.h5")
            else :
                # use pre-trained GRU model with embedding layer
                self.clf[dsid] = load_model("gru_conceptNet.h5")

        # loading tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # vectorization
        sequence = tokenizer.texts_to_sequences([vals])

        # padding sequence
        pading_sequence = pad_sequences(sequence,maxlen = 35)

        # make prediction
        predScore = self.clf[dsid].predict(pading_sequence).flatten()
        predLabel = int(np.around(predScore)[0])

        self.write_json({"prediction":str(predLabel)})
