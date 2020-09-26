#!-*-coding:utf-8-*-
import os
import sys
import json
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import japanize_matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.tree import plot_tree
from .tools.logger import *
from .tools.file_modules import *
from abc import ABCMeta , abstractmethod
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


__all__ = ["decisiontree"]

class Prediction(object,metaclass=ABCMeta):
    @abstractmethod
    def predict( self ):
        """
        推定メソッド
        --------------------------------
        """
        raise NotImplementedError()

    @abstractmethod
    def accuracy( self ):
        """
        精度メソッド
        --------------------------------
        """
        raise NotImplementedError()

class decisiontree(Prediction):
    def __init__( self , df , cols , filename ):
        self.df = df
        self.filename = filename
        self.x_cols = json.loads( cols["x"] )
        self.y_cols = json.loads( cols["y"] )
        self.query = cols["query"]
        self.save = bool( cols["save"] )
        self.max_depth = cols["max_depth"]

        if self.max_depth == "None":
            self.max_depth = None
        else:
            self.max_depth = int( self.max_depth )

        datetime_columns_arr = datetime_colmuns( df=self.df )
        for col in datetime_columns_arr:
            self.df[col] = pd.to_datetime( self.df[col] ).dt.strftime("%Y-%m-%d")

    def predict( self ):
        df = self.df.query( self.query )
        X = df[self.x_cols]
        Y = df[self.y_cols]

        X_train , X_test , Y_train , Y_test = train_test_split( X , Y )

        model = DecisionTreeClassifier(max_depth=self.max_depth)
        model.fit( X_train , Y_train )

        if self.save is True:
            # Plot Decision Tree
            self.__tree_plot( model=model , X=X_train , Y=Y_train )

            # Plot feature importance
            self.__importance_plot( model=model )

        return model

    def accuracy( self , model ):
        df = self.df.query( self.query )
        X = df[self.x_cols]
        Y = df[self.y_cols]

        X_train , X_test , Y_train , Y_test = train_test_split( X , Y )
        Y_test = np.array( Y_test ).T[0]
        predicted = model.predict( X_test )
        predicted_train = model.predict( X_train )

        N = len( predicted ) + len( predicted_train )

        train_acc = round( sum( predicted_train ==\
         np.array(Y_train).T[0] ) / len( predicted_train ) , 3 )

        test_acc = round( sum( predicted == Y_test ) / len( Y_test ) , 3 )

        return { "N" : N , "train" : train_acc , "test" : test_acc }

    def __tree_plot( self , model , X , Y ):
        import matplotlib as mpl
        mpl.rcParams.update(mpl.rcParamsDefault)
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot()
        plot_tree( model , filled=True )

        filename = os.path.dirname( self.filename.replace( "datas" , "graphs" ) )
        filename = filename + "/decisiontree_{}_{}.png".\
        format( self.y_cols[0] , "_".join( self.x_cols ) )
        if os.path.exists( os.path.dirname( filename ) ) is not True:
            message( "mkdir {}".format( os.path.dirname( filename ) ) )
            os.makedirs( os.path.dirname( filename ) )

        message( "saved Decision Tree image as {}".format( filename ) )
        plt.savefig( filename )

    def __importance_plot( self , model ):
        sns.set()
        sns.set(font='Yu Gothic')
        sns.set( font=["IPAexGothic"], font_scale=0.8 )
        plt.cla()
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.xlim( 0 , 1 )
        filename = os.path.dirname( self.filename.replace( "datas" , "graphs" ) )
        filename = filename + "/decisiontree_{}_{}_importance.png".\
        format( self.y_cols[0] , "_".join( self.x_cols ) )
        plt.barh( self.x_cols , model.feature_importances_ )
        message( "saved Decision Tree feature importance image as {}".format( filename ) )
        plt.savefig( filename )
