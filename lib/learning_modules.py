#!-*-coding:utf-8-*-
import os
import sys
import json
import pickle as pkl
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import japanize_matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.tree import plot_tree
from .tools.logger import *
from .tools.file_modules import *
from abc import ABCMeta , abstractmethod
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split


__all__ = ["decisiontree","logistic"]


class Learning(object,metaclass=ABCMeta):
    """
    学習オブジェクトの抽象クラス
    """
    @abstractmethod
    def learn( self ):
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

    @abstractmethod
    def dump( self ):
        """
        モデルを保存する
        --------------------------------
        """

        raise NotImplementedError()

class LearnController:
    """
    学習クラスの親クラス
    """
    def __init__( self ):
        pass

    def learning_set( self , model , df , query , x_cols , y_cols ):
        if query != "":
            df = df.query( query )

        df = df.query( query )
        X = df[x_cols]
        Y = df[y_cols]

        X_train , X_test , Y_train , Y_test = train_test_split( X , Y )

        model.fit( X_train , Y_train )

        return model


    def acc_calc( self , model , df , query , x_cols , y_cols ):
        if query != "":
            df = df.query( query )

        X = df[x_cols]
        Y = df[y_cols]

        X_train , X_test , Y_train , Y_test = train_test_split( X , Y )
        Y_test = np.array( Y_test ).T[0]
        predicted = model.predict( X_test )
        predicted_train = model.predict( X_train )

        N = len( predicted ) + len( predicted_train )

        train_acc = round( sum( predicted_train ==\
         np.array(Y_train).T[0] ) / len( predicted_train ) , 3 )

        test_acc = round( sum( predicted == Y_test ) / len( Y_test ) , 3 )

        return { "N" : N , "train" : train_acc , "test" : test_acc }

    def model_save( self , model , filename , modelname , x_cols , y_cols ):
        filename = os.path.dirname( filename.replace( "datas" , "models" ) )
        filename = filename.replace("/shaped","")
        filename = filename.replace("/originals","")
        filename = filename.replace("/statistics","")
        filename = filename + "/{}_model_{}_{}.sav".\
        format( modelname , y_cols[0] , "_".join( x_cols ) )

        if os.path.exists( os.path.dirname( filename ) ) is not True:
            message( "mkdir {}".format( os.path.dirname( filename ) ) )
            os.makedirs( os.path.dirname( filename ) )

        pkl.dump( model , open( filename , "wb" ) )
        message( "{} tree model dumped as {}".format( modelname , filename ) )


class decisiontree(Learning,LearnController):
    def __init__( self , df , cols , filename ):
        super(decisiontree, self).__init__()
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

    def learn( self ):
        model = self.learning_set( model=DecisionTreeClassifier(max_depth=self.max_depth) ,\
         df=self.df , query=self.query ,x_cols=self.x_cols , y_cols=self.y_cols )

        if self.save is True:
            # Plot Decision Tree
            self.__tree_plot( model=model )

            # Plot feature importance
            self.__importance_plot( model=model )

        return model

    def accuracy( self , model ):
        report_dict = self.acc_calc( model=model , df=self.df , query=self.query ,\
         x_cols=self.x_cols , y_cols=self.y_cols )

        return report_dict

    def dump( self , model ):
        self.model_save( model=model , filename=self.filename ,\
         modelname="decisiontree" , x_cols=self.x_cols , y_cols=self.y_cols )


    def __tree_plot( self , model ):
        mpl.rcParams.update(mpl.rcParamsDefault)
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot()
        plot_tree( model , filled=True )

        filename = os.path.dirname( self.filename.replace( "datas" , "graphs" ) )
        filename = filename + "/decisiontree_depth{}_{}_{}.png".\
        format( self.max_depth , self.y_cols[0] , "_".join( self.x_cols ) )
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
        filename = filename + "/decisiontree_depth{}_{}_{}_importance.png".\
        format( self.max_depth , self.y_cols[0] , "_".join( self.x_cols ) )
        plt.barh( self.x_cols , model.feature_importances_ )
        message( "saved Decision Tree feature importance image as {}".format( filename ) )
        plt.savefig( filename )


class logistic(Learning,LearnController):
    def __init__( self , df , cols , filename ):
        super(logistic, self).__init__()
        self.df = df
        self.filename = filename
        self.x_cols = json.loads( cols["x"] )
        self.y_cols = json.loads( cols["y"] )
        self.query = cols["query"]
        self.save = bool( cols["save"] )

        datetime_columns_arr = datetime_colmuns( df=self.df )
        for col in datetime_columns_arr:
            self.df[col] = pd.to_datetime( self.df[col] ).dt.strftime("%Y-%m-%d")

    def learn( self ):
        model = self.learning_set( model=LogisticRegression() ,\
         df=self.df , query=self.query ,x_cols=self.x_cols , y_cols=self.y_cols )

        return model


    def accuracy( self , model ):
        report_dict = self.acc_calc( model=model , df=self.df , query=self.query ,\
         x_cols=self.x_cols , y_cols=self.y_cols )

        self.__plot_calibration( df=self.df , query=self.query , model=model ,\
         x_cols=self.x_cols , y_cols=self.y_cols )

        return report_dict

    def dump( self , model ):
        self.model_save( model=model , filename=self.filename ,\
         modelname="logistic" , x_cols=self.x_cols , y_cols=self.y_cols )

    def __plot_calibration( self , df , query , model , x_cols , y_cols ):
        if query != "":
            df = df.query( query )

        X = df[x_cols]
        Y = df[y_cols]
        X_train , X_test , Y_train , Y_test = train_test_split( X , Y )

        N = len( Y_test )
        bins_num = int( ( ( N / N ** 0.5 ) * 0.5 ) * 2 )

        prob = model.predict_proba( X_test )[:,1]
        prob_true , prob_pred = calibration_curve( y_true=Y_test ,\
         y_prob=prob , n_bins=bins_num )

        fig = plt.figure()
        ax1 = plt.subplot(2,1,1)
        ax1.set_title("TEST SAMPLE NUM={}".format(N))
        ax1.plot( prob_pred , prob_true , marker="s" , label="calibration_curve" )
        ax1.plot( [0,1],[0,1],linestyle="--",label="ideal" )
        ax1.legend()

        ax2 = plt.subplot(2,1,2)
        ax2.hist( prob , bins=40 , histtype="step" )
        ax2.set_xlim(0,1)
        filename = os.path.dirname( self.filename.replace( "datas" , "graphs" ) )
        filename = filename + "/calibration_curve{}_{}.png".\
        format( self.y_cols[0] , "_".join( self.x_cols ) )
        if os.path.exists( os.path.dirname( filename ) ) is not True:
            message( "mkdir {}".format( os.path.dirname( filename ) ) )
            os.makedirs( os.path.dirname( filename ) )

        message( "saved calibration_curve image as {}".format( filename ) )
        plt.savefig( filename )
