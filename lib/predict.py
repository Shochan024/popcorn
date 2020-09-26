#!-*-coding:utf-8-*-
import os
import json
import pydotplus
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta , abstractmethod
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from .tools.logger import *

__all__ = ["decisiontree"]

class Prediction(object,metaclass=ABCMeta):
    @abstractmethod
    def predict( self ):
        """
        推定メソッド
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

    def predict( self ):
        X = self.df[self.x_cols]
        Y = self.df[self.y_cols]
        model = DecisionTreeClassifier()
        model.fit( X , Y )

        if self.save is True:
            fig = plt.figure()
            ax = fig.add_subplot()
            plot_tree( model , feature_names=self.x_cols , ax=ax ,\
             class_names=np.unique( Y ).astype(np.str) , filled=True)

            filename = os.path.dirname( self.filename.replace( "datas" , "graphs" ) )
            filename = filename + "/decisiontree_{}_{}.png".\
            format( self.y_cols[0] , "_".join( self.x_cols ) )
            if os.path.exists( os.path.dirname( filename ) ) is not True:
                system( "mkdir {}".format( os.path.dirname( filename ) ) )
                os.makedirs( os.path.dirname( filename ) )

            system( "saved Decision Tree image as {}".format( filename ) )
            plt.savefig( filename )

        return model
