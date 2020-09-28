#!-*-coding:utf-8-*-
import os
import sys
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from abc import ABCMeta , abstractmethod
from sklearn.datasets import make_classification
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from .tools.logger import *

__all__ = ["logisticRegression"]

class Simurater(object,metaclass=ABCMeta):
    @abstractmethod
    def performance( self ):
        """
        modelの性能を測るメソッド
        """
        raise NotImplementedError()


class logisticRegression(Simurater):
    def __init__( self , options , workdir ):
        self.options = options
        self.workdir = workdir

    def performance( self ):
        sns.set()
        for how , property in self.options.items():
            how = how.split("_")[0]
            model = LogisticRegression()
            result = eval( "self._{}".format( how ) )( model=model , property=property )
            filename = result["filename"]
            if os.path.exists( os.path.dirname( filename ) ) is not True:
                message( "mkdir {}".format( os.path.dirname( filename ) ) )
                os.makedirs( os.path.dirname( filename ) )

            result["fig"].save( filename )
            message( "saved calibration_curve image as {}".format( filename ) )

    def _calibration( self , model , property ):
        N = json.loads( property["N"] )
        probability = float( json.loads( property["detail"]["probability"] ) )
        n_features = json.loads( property["detail"]["n_features"] )
        n_informative = int( json.loads( property["detail"]["n_informative"] ) )
        n_redundant = int( json.loads( property["detail"]["n_redundant"] ) )
        frames = np.arange( n_features+n_informative*2,N,int( max(1,( N**0.5 )  ) ) )

        filename = "{}logistic/{}_N{}_probability{}_informative{}_n_features{}_n_redundant{}.gif".\
        format( self.workdir ,\
         "calibration" , N , probability , n_informative , n_features , n_redundant )

        fig = plt.figure()
        ani = animation.FuncAnimation(fig,self._update,\
        fargs=([n_features,n_informative,n_redundant,probability],),frames=frames ,interval=100 )

        return { "filename" : filename , "fig" : ani }

    def _update( self , n , details ):
        model = LogisticRegression(random_state=0)
        X, Y = make_classification( n_samples=n,scale=100,\
         n_features=details[0], n_informative=details[1],n_classes=2,weights=[1-details[3],details[3] ],\
          n_redundant=details[2],random_state=0,shuffle=False)

        X_train , X_test , Y_train , Y_test = train_test_split( X , Y )
        # すべての要素が同じだとエラーになるので、まだらになるまで分割を繰り返す
        if len( np.unique( Y_train ) ) < 2:
            while len( np.unique( Y_train ) ) < 2:
                X_train , X_test , Y_train , Y_test = train_test_split( X , Y )

        model.fit( X_train , Y_train  )
        N = len( Y_test )
        bins_num = max( 10 , int( ( ( N / N ** 0.5 ) * 0.5 ) * 2 ) )

        prob = model.predict_proba( X_test )[:,1]
        prob_true , prob_pred = calibration_curve( y_true=Y_test ,\
        y_prob=prob , n_bins=bins_num )

        ax1 = plt.subplot(2,1,1)
        ax1.cla()
        ax1.set_title( "FEATURE={},PROB={},SAMPLE NUM={}".format( details[0],details[3] , n ) )
        ax1.plot( prob_pred , prob_true , marker="s" , label="calibration_curve" )
        ax1.plot( [0,1],[0,1],linestyle="--",label="ideal" )

        ax2 = plt.subplot(2,1,2)
        ax2.cla()
        ax2.hist( prob , bins=40 , histtype="step" )
        ax2.set_xlim(0,1)
