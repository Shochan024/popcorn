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
from abc import ABCMeta , abstractmethod
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from .tools.logger import *
from .tools.file_modules import *


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
        model = DecisionTreeClassifier(max_depth=self.max_depth)
        model.fit( X , Y )

        if self.save is True:
            # Plot Decision Tree
            self.__tree_plot( model=model , X=X , Y=Y )

            # Plot feature importance
            self.__importance_plot( model=model )

        return model

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
            system( "mkdir {}".format( os.path.dirname( filename ) ) )
            os.makedirs( os.path.dirname( filename ) )

        system( "saved Decision Tree image as {}".format( filename ) )
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
        system( "saved Decision Tree feature importance image as {}".format( filename ) )
        plt.savefig( filename )
