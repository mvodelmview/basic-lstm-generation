# Author: Nick Emerson 
# Last Edit: November 20, 2021
# This is a program for initializing and training RNN model plus text generation



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os


class ModelLSTM:
    
    
    def __init__(self, text):
        '''
        Input: single string of text (as a python string). Could be section of novels, concatenated product reviews, or something else
        
        '''
        print("Vectorizing text sequence...")
        self.subseq_length = 90
    
        step = 3

        self.unique_chars = sorted(list(set(text)))
        self.char_indices = dict((char, self.unique_chars.index(char)) for char in self.unique_chars)
        
        # model takes in preceding sequence of subseq_length characters
        # and is trained to predict the following target char at the subseq_length + 1 position
        raw_subseqs = []
        raw_target_chars = []

        for i in range(0, len(text) - self.subseq_length, step):
            raw_subseqs.append(text[i : i + self.subseq_length])
            raw_target_chars.append(text[i + self.subseq_length])

        # one-hot encoding
        self.subseqs = np.zeros((len(raw_subseqs), self.subseq_length, len(self.unique_chars)), dtype=np.bool)
        self.target_chars = np.zeros((len(raw_subseqs), len(self.unique_chars)), dtype=np.bool)
        
        for subseq_index, subseq in enumerate(raw_subseqs):
            self.target_chars[subseq_index, self.char_indices[raw_target_chars[subseq_index]]] = 1
            for char_index, char in enumerate(subseq):
                self.subseqs[subseq_index, char_index, self.char_indices[char]] = 1
            
        print("Text vectorization done. Compiling model...")
        self.init_model()
          
            
    def init_model(self):
        '''
        helper function called by constructor, creates and compiles Keras single layer LSTM model
        
        '''
        self.model = keras.models.Sequential()
        self.model.add(layers.LSTM(128, input_shape=(self.subseq_length, len(self.unique_chars))))
        self.model.add(layers.Dense(len(self.unique_chars), activation='softmax'))
        optimizer = keras.optimizers.RMSprop(lr=0.01)

        self.callbacks = [
            keras.callbacks.EarlyStopping(monitor='acc', patience=4),
            keras.callbacks.ModelCheckpoint(filepath="my_models/apoc_best.h5", monitor='loss', save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor='loss', factor= 0.5, patience=1)]
        
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        
        print("Model compilation finished. Call yourInstanceHere.train() to train the model.")
                
        
    def train(self, epochs=100):
        '''
        method to train the model, separated from compilation method (init_model) because training takes a significant amount of time
        
        '''
        self.model.fit(self.subseqs, self.target_chars, batch_size=128, callbacks=self.callbacks, epochs=epochs)
        
        
    def sample_index(self, prediction, randomness):
        '''
        helper method, receives probability distribution of character index ("prediction") and randomness scalar which makes 
        predictions more random (and interesting)
        
        '''
        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction) / randomness
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        return np.argmax(np.random.multinomial(1, prediction, 1))
    
    
    def generate_content(self, text_seed, randomness=0.4, content_length=400):
        '''
        method generates around a paragraph of content
        text_seed is a string (as a python string) which is used as a seed. 
  
        then this method recursively predicts the following character based off of 
        current characters in text_seed. It then appends this character and predicts the next character. 
        
        output is paragraph of generated text based off of training data provided as text arg to the constructor,
        preceded by the input text_seed

        content_length is the number of characters the caller wants to generate in addition to the text_seed
        
        '''
        text_seed = text_seed.lower()
        original_text_seed = text_seed
        
        review = ""
        i = 1

        while i <= 400:
            seed_sample = np.zeros((1, len(text_seed), len(self.unique_chars)))
            for i, char_index in enumerate(text_seed):
                seed_sample[0, i, self.char_indices[char_index]] = 1

            prediction = self.model.predict(seed_sample, verbose=0)[0]

            new_char = self.unique_chars[self.sample_index(prediction, randomness=randomness)]

            text_seed += new_char
            text_seed = text_seed[1:]

            review = review + new_char
            i += 1
            
        return original_text_seed + review
    
    
    def save_model(self, file_name):
        '''
        saves current model in models dir in the same directory
      
        '''
        if not os.path.exists("models"):
            os.makedirs("models")
        self.model.save("models/" + file_name)
        
        
