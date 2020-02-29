"""
Training the Sentence Similarity Predictor using a Wide NN
"""

import os
import sys
import re
import pickle
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D, concatenate, dot
from keras.models import Model, Sequential, load_model
from keras.utils import plot_model
import warnings
from imp import reload
warnings.filterwarnings('ignore')
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")


main_dir = os.path.dirname(os.path.dirname(__file__))
training_data = os.path.join(main_dir, "data", "sentsim", "questions.csv")
model_hdf = os.path.join(main_dir, "models", "sentsim", "widenn-model.hdf5")
tokenizer_pkl = os.path.join(main_dir, "models", "sentsim", "tokenizer.pkl")
model_image = os.path.join(main_dir, "assets", "widenn.png")
stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

class WideNNTrainer:
    def __init__(self, training_csv, store_model, store_tokenizer, max_features=6000, maxlen=100):
        """
        Set location of the training csv and where to store the mode
        """
        self.df = pd.read_csv(training_csv)
        self.store_model = store_model
        self.max_features = max_features
        # padded length of the question
        self.maxlen=100
        self.model = None
        if os.path.exists(tokenizer_pkl):
            self.tokenizer = pickle.load(open(tokenizer_pkl, "rb"))
        else:
            self.tokenizer = Tokenizer(num_words=self.max_features)
        self.process_data()

    def model_creator(self):
        """
        Model Architecture
        """
        embedding_size = 128

        inp1 = Input(shape=(self.maxlen,))
        inp2 = Input(shape=(self.maxlen,))

        x1 = Embedding(self.max_features, embedding_size)(inp1)
        x2 = Embedding(self.max_features, embedding_size)(inp2)

        x3 = Bidirectional(LSTM(32, return_sequences = True))(x1)
        x4 = Bidirectional(LSTM(32, return_sequences = True))(x2)

        x5 = GlobalMaxPool1D()(x3)
        x6 = GlobalMaxPool1D()(x4)

        x7 =  dot([x5, x6], axes=1)

        x8 = Dense(40, activation='relu')(x7)
        x9 = Dropout(0.05)(x8)
        x10 = Dense(10, activation='relu')(x9)
        output = Dense(1, activation="sigmoid")(x10)

        return Model(inputs=[inp1, inp2], outputs=output)
    
    def process_data(self):
        """
        Data Preprocessing
        """
        # drop all id columns
        self.df.drop(['id', 'qid1', 'qid2'], axis=1)
        for i in ('question1', 'question2'):
            self.df[i] = self.df[i].apply(lambda x: clean_text(str(x)))
        total_text = pd.concat([self.df['question1'], self.df['question2']]).reset_index(drop=True)
        if not os.path.exists(tokenizer_pkl):
            self.tokenizer.fit_on_texts(total_text)
            pickle.dump(self.tokenizer, open(tokenizer_pkl, "wb"))
        self.x = []
        self.y = self.df['is_duplicate']
        for i in ('question1', 'question2'):
            # use tokenizer to convert text to sequences and then pad the sequences to a max of 100
            self.x.append(
                            pad_sequences(
                                self.tokenizer.texts_to_sequences(
                                    self.df[i]
                                ), 
                                maxlen=self.maxlen
                            )
                        )
        self.model = self.model_creator()
    
    def train(self):
        """
        Train the model
        """
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        batch_size = 100
        epochs = 3
        self.model.fit(self.x, 
                    self.y, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_split=0.2)
        self.model.save(model_hdf)
    
    def predict(self, sentence_1, sentence_2, threshold=0.15):
        """
        Predict using saved model
        """
        model = load_model(model_hdf)
        plot_model(model, to_file=model_image)
        sentence_1 = clean_text(sentence_1)
        sentence_2 = clean_text(sentence_2)
        sentence_sequenced = self.tokenizer.texts_to_sequences([sentence_1, sentence_2])
        sentence_padded = pad_sequences(sentence_sequenced, self.maxlen)
        sentence1_padded = sentence_padded[0:1]
        sentence2_padded = sentence_padded[1:]
        prediction = model.predict([sentence1_padded, sentence2_padded])
        if prediction[0][0] >= threshold:
            return True
        return False

    


if __name__=="__main__":
    # test_cases = [
    #     ("Can I get a burrito from Spain", "If I am in Spain, can I get a burrito"),
    #     ("I am angry", "I am livid"),
    #     ("I am not very sad", "I am normal")
    #     ]
    
    trainer = WideNNTrainer(training_data, model_hdf, tokenizer_pkl)
    if not os.path.exists(model_hdf):
        trainer.train()
    if len(sys.argv) == 3:
        print("First sentence: ", sys.argv[1], "\n")
        print ("Second sentence: ", sys.argv[2], "\n")
        print("Similar? : ", trainer.predict(sys.argv[1], sys.argv[2]),"\n")