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
from scipy.stats import sem
from sklearn.svm import SVC
from sklearn.tree import plot_tree
from .tools.logger import *
from .tools.file_modules import *
from lifelines import KaplanMeierFitter
from abc import ABCMeta , abstractmethod
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split , cross_val_score , KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score , roc_curve, auc
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, ExpSineSquared, Matern, RationalQuadratic
from sklearn.preprocessing import StandardScaler

dpi = 200
plt.rcParams["figure.dpi"] = dpi

__all__ = ["decisiontree","logistic","svm","randomforest"]
__all__ += ["kaplanmeierfitter","gausprocess"]


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

    def learning_set( self , model , df , query , x_cols , y_cols , std ):
        if query != "":
            df = df.query( query )

        df = df.query( query )
        X = df[x_cols].astype("float64")
        Y = df[y_cols].astype("float64")

        X = self._to_norm( cols=std , X=X )

        X_train , X_test , Y_train , Y_test = train_test_split( X , Y )

        model.fit( X_train , Y_train )

        return model

    def acc_calc( self , model , df , query , x_cols , y_cols , std ):
        if query != "":
            df = df.query( query )

        X = df[x_cols]
        Y = df[y_cols]

        X = self._to_norm( cols=std , X=X )

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


    def _plot_calibration( self , df , query , model , x_cols , y_cols , std ):
        if query != "":
            df = df.query( query )

        X = df[x_cols]
        Y = df[y_cols]

        X = self._to_norm( cols=std , X=X )

        X_train , X_test , Y_train , Y_test = train_test_split( X , Y )

        N = len( Y_test )
        bins_num = int( ( ( N / N ** 0.5 ) * 0.5 ) * 2 )

        pred = model.predict( X_test )
        pred = round( sum( pred ) / len( pred ) , 3 )
        system( " {} 1:{},0:{}".format( str( model ) , pred , 1-pred ) )


        prob = model.predict_proba( X_test )[:,1]
        prob_true , prob_pred = calibration_curve( y_true=Y_test ,\
         y_prob=prob , n_bins=bins_num )


        probablility = round( np.sum(np.array(Y_test).astype(np.int)) / len(Y_test) , 3 )

        plt.clf()
        plt.rcParams["figure.dpi"] = dpi
        fig = plt.figure()
        ax1 = plt.subplot(2,1,1)
        ax1.set_title("FEATURE={},PROB={},TEST SAMPLE NUM={}\n std:{}".format(X.shape[1],probablility,N,len(std)!=0))
        ax1.plot( prob_pred , prob_true , marker="s" , label="calibration_curve" )
        ax1.plot( [0,1],[0,1],linestyle="--",label="ideal" )
        ax1.legend()

        ax2 = plt.subplot(2,1,2)
        ax2.hist( prob , bins=40 , histtype="step" )
        ax2.set_xlim(0,1)

        val_names = "_".join( self.x_cols )

        filename = os.path.dirname( self.filename.replace( "datas" , "graphs" ) )
        filename = filename + "/Calibration/{}/{}_calibration_curve{}_std{}.png".\
        format( val_names , str( model ) , self.y_cols[0] , len( std ) != 0 )
        if os.path.exists( os.path.dirname( filename ) ) is not True:
            message( "mkdir {}".format( os.path.dirname( filename ) ) )
            os.makedirs( os.path.dirname( filename ) )

        message( "saved calibration_curve image as {}".format( filename ) )
        plt.savefig( filename )


    def _evaluation_cross_validation( self , model , x , y , k ):
        cv = KFold( n_splits=k , random_state=0 , shuffle=False )
        score = cross_val_score( model , x , y , cv=cv )


        return "Mean Score: {} (+/-{})".format( np.mean( score ) , sem( score ) )


    def _plot_spec( self , df , query , model , x_cols , y_cols , std ):
        if query != "":
            df = df.query( query )


        X = df[x_cols]
        Y = df[y_cols]

        X = self._to_norm( cols=std , X=X )

        X_train , X_test , Y_train , Y_test = train_test_split( X , Y )

        N = len( Y_test )
        bins_num = int( ( ( N / N ** 0.5 ) * 0.5 ) * 2 )

        Y_pred = model.predict( X_test )
        confusion = confusion_matrix( Y_test , Y_pred )
        precision = round( precision_score( Y_test , Y_pred ) , 3 )
        recall = round( recall_score( Y_test , Y_pred ) , 3 )
        f1 = round( f1_score( Y_test , Y_pred ) , 3 )
        labels = list( set( Y_test ) )
        confusion_df = pd.DataFrame( confusion )

        cross_val_score = self._evaluation_cross_validation( model=model , x=X_train , y=Y_train , k=3 )

        fpr, tpr, thresholds = roc_curve( y_true = Y_test , y_score=Y_pred )

        plt.clf()
        plt.rcParams["figure.dpi"] = dpi
        plt.title( "{} \n precision:{} recall:{} f1:{} std:{}".format( str( model ) , \
        precision , recall , f1 , len( std ) !=0 ) )
        plt.plot(fpr, tpr, label='roc curve (area = %0.3f)' % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], linestyle='--', label='random')
        plt.plot([0, 0, 1], [0, 1, 1], linestyle='--', label='ideal')
        plt.legend()
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')

        val_names = "_".join( self.x_cols )
        filename = os.path.dirname( self.filename.replace( "datas" , "graphs" ) )
        confusion_heatmap = filename + "/Confusion/{}/{}_Confusion{}_std_{}.png".\
        format( val_names , str( model ) , self.y_cols[0] , len( std ) != 0 )

        filename = filename + "/ROC/{}/{}_ROC_curve{}_std_{}.png".\
        format( val_names , str( model ) , self.y_cols[0] , len( std ) != 0 )

        message( "saved ROC_curve image as {}".format( filename ) )
        if os.path.exists( os.path.dirname( filename ) ) is not True:
            os.makedirs( os.path.dirname( filename ) )

        plt.savefig( filename )

        plt.clf()
        sns.heatmap( confusion_df , vmin=0 , vmax=100 , fmt="d" , annot_kws={"fontsize": 20} , cmap="Blues" , annot=True )
        message( "saved Confusion image as {}".format( confusion_heatmap ) )
        if os.path.exists( os.path.dirname( confusion_heatmap ) ) is not True:
            os.makedirs( os.path.dirname( confusion_heatmap ) )

        plt.savefig( confusion_heatmap )

        print("\n")
        system( "{}".format( str( " : ".join( self.x_cols ) ) ) )
        system( "{} Confusion Matrix \n{}".format( str( model ) , confusion ) )
        system( "{} Precision  {}".format( str( model ) , precision ) )
        system( "{} Recall  {}".format( str( model ) , recall ) )
        system( "{} F1 {}".format( str( model ) , f1 ) )
        system( "{} Cross Val Score {}".format( str( model ) ,\
         cross_val_score ) )

    def _to_norm( self , cols , X ):
        for col in cols:
            X[col] = ( X[col] - X[col].mean() ) / X[col].std()

        return X


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
        self.std = json.loads( cols["std"] )

        if self.max_depth == "None":
            self.max_depth = None
        else:
            self.max_depth = int( self.max_depth )

        datetime_columns_arr = datetime_colmuns( df=self.df )
        for col in datetime_columns_arr:
            self.df[col] = pd.to_datetime( self.df[col] ).dt.strftime("%Y-%m-%d")

    def learn( self ):
        model = self.learning_set( model=DecisionTreeClassifier(max_depth=self.max_depth) ,\
         df=self.df , query=self.query ,x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        if self.save is True:
            # Plot Decision Tree
            self.__tree_plot( model=model , df=self.df ,\
             query=self.query , x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

            # Plot feature importance
            self.__importance_plot( model=model )

        return model

    def accuracy( self , model ):
        report_dict = self.acc_calc( model=model , df=self.df , query=self.query ,\
         x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        self._plot_calibration( df=self.df , query=self.query , model=model ,\
         x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        self._plot_spec( df=self.df , query=self.query , model=model , \
        x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        return report_dict

    def dump( self , model ):
        self.model_save( model=model , filename=self.filename ,\
         modelname="decisiontree" , x_cols=self.x_cols , y_cols=self.y_cols )



    def __tree_plot( self , model , df , query , x_cols , y_cols , std ):
        if query != "":
            df = df.query( query )

        X = df[x_cols]
        Y = df[y_cols]

        X = self._to_norm( cols=std , X=X )

        X_train , X_test , Y_train , Y_test = train_test_split( X , Y )

        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams["font.family"] = "IPAexGothic"
        plt.rcParams["figure.dpi"] = dpi
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot()
        plot_tree( model , filled=True , feature_names=x_cols )

        pred = model.predict( X_test )
        pred = round( sum( pred ) / len( pred ) , 3 )
        system( "1:{},0:{}".format( pred , 1-pred ) )

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
        sns.set( font=["IPAexGothic"], font_scale=0.4 )
        plt.cla()
        plt.rcParams["figure.dpi"] = dpi
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
        self.C = float( cols["C"] )
        self.query = cols["query"]
        self.save = bool( cols["save"] )
        self.std = json.loads( cols["std"] )

        datetime_columns_arr = datetime_colmuns( df=self.df )
        for col in datetime_columns_arr:
            self.df[col] = pd.to_datetime( self.df[col] ).dt.strftime("%Y-%m-%d")

    def learn( self ):
        model = self.learning_set( model=LogisticRegression(C=self.C) ,\
         df=self.df , query=self.query ,x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        return model


    def accuracy( self , model ):
        report_dict = self.acc_calc( model=model , df=self.df , query=self.query ,\
         x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        self._plot_calibration( df=self.df , query=self.query , model=model ,\
         x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        self._plot_spec( df=self.df , query=self.query , model=model , \
        x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        self.__coef_plot( model=model , x_cols=self.x_cols )

        return report_dict

    def dump( self , model ):
        self.model_save( model=model , filename=self.filename ,\
         modelname="logistic" , x_cols=self.x_cols , y_cols=self.y_cols )

    def __coef_plot( self , model , x_cols ):
        plt.clf()
        plt.rcParams["figure.dpi"] = dpi
        plt.title( "{} coef".format( str( model ) ) )
        plt.bar( x_cols , model.coef_[0] )
        filename = os.path.dirname( self.filename.replace( "datas" , "graphs" ) )
        filename = filename + "/{}_{}_coef.png".format( model , "_".join( self.x_cols ) )
        system( "{} coef : {}".format( str( model ) , str( model.coef_ ) ) )
        message( "saved Coef image as {}".format( filename ) )
        plt.savefig( filename )


class svm(Learning,LearnController):
    def __init__( self , df , cols , filename ):
        super(svm, self).__init__()
        self.df = df
        self.filename = filename
        self.x_cols = json.loads( cols["x"] )
        self.y_cols = json.loads( cols["y"] )
        self.query = cols["query"]
        self.kernel = cols["kernel"]
        self.save = bool( cols["save"] )
        self.std = json.loads( cols["std"] )

        datetime_columns_arr = datetime_colmuns( df=self.df )
        for col in datetime_columns_arr:
            self.df[col] = pd.to_datetime( self.df[col] ).dt.strftime("%Y-%m-%d")

    def learn( self ):
        model = self.learning_set( model=SVC(probability=True,\
        kernel=self.kernel,gamma="auto",random_state=None) ,\
         df=self.df , query=self.query ,x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        return model


    def accuracy( self , model ):
        report_dict = self.acc_calc( model=model , df=self.df , query=self.query ,\
         x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        self._plot_calibration( df=self.df , query=self.query , model=model ,\
         x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        self._plot_spec( df=self.df , query=self.query , model=model , \
        x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        return report_dict

    def dump( self , model ):
        self.model_save( model=model , filename=self.filename ,\
         modelname="svm" , x_cols=self.x_cols , y_cols=self.y_cols )


class randomforest(Learning,LearnController):
    def __init__( self , df , cols , filename ):
        super(randomforest, self).__init__()
        self.df = df
        self.filename = filename
        self.x_cols = json.loads( cols["x"] )
        self.y_cols = json.loads( cols["y"] )
        self.query = cols["query"]
        self.save = bool( cols["save"] )
        self.max_depth = cols["max_depth"]
        self.std = json.loads( cols["std"] )
        self.criterion = cols["criterion"]

        if self.max_depth == "None":
            self.max_depth = None
        else:
            self.max_depth = int( self.max_depth )

        datetime_columns_arr = datetime_colmuns( df=self.df )
        for col in datetime_columns_arr:
            self.df[col] = pd.to_datetime( self.df[col] ).dt.strftime("%Y-%m-%d")

    def learn( self ):
        model = self.learning_set( model=RandomForestClassifier(criterion=self.criterion,max_depth=self.max_depth) ,\
         df=self.df , query=self.query ,x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        return model

    def accuracy( self , model ):
        report_dict = self.acc_calc( model=model , df=self.df , query=self.query ,\
         x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        self._plot_calibration( df=self.df , query=self.query , model=model ,\
         x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        self._plot_spec( df=self.df , query=self.query , model=model , \
        x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        return report_dict

    def dump( self , model ):
        self.model_save( model=model , filename=self.filename ,\
         modelname="randomforest" , x_cols=self.x_cols , y_cols=self.y_cols )


class kaplanmeierfitter(Learning,LearnController):
    def __init__( self , df , cols , filename ):
        super(kaplanmeierfitter, self).__init__()
        self.df = df
        self.filename = filename
        self.x_cols = json.loads( cols["x"] )
        self.y_cols = json.loads( cols["y"] )
        self.query = cols["query"]
        self.save = bool( cols["save"] )

    def learn( self ):
        model = self.learning_set( model=KaplanMeierFitter() ,\
         df=self.df , query=self.query ,x_cols=self.x_cols , y_cols=self.y_cols , std=[] )

        return model

    def accuracy( self , model ):
        report_dict = { "N" : [] , "train" : [] , "test" : [] }

        return report_dict

    def dump( self , model ):
        self.__lifetime_plot( model=model )
        self.model_save( model=model , filename=self.filename ,\
         modelname="kaplanmeierfitter" , x_cols=self.x_cols , y_cols=self.y_cols )

    def __lifetime_plot( self , model ):
            filename = os.path.dirname( self.filename.replace( "datas" , "graphs" ) )
            filename = filename + "/{}_{}_{}_lifetime.png".format( "kaplanmeierfitter" , "-".join( self.y_cols ) ,\
                 "-".join( self.x_cols ) )

            plt.clf()
            ax = model.plot()
            ax.set_ylim( 0 , 1 )
            ax.get_figure().savefig( filename )

class gausprocess(Learning,LearnController):

    def __init__( self , df , cols , filename ):
        super(gausprocess, self).__init__()
        self.df = df
        self.filename = filename
        self.x_cols = json.loads( cols["x"] )
        self.y_cols = json.loads( cols["y"] )
        self.std = json.loads( cols["std"] )
        self.query = cols["query"]
        self.save = bool( cols["save"] )
        self.kernel = cols["kernel"]

    def learn( self ):
        model = self.learning_set( model=\
        GaussianProcessRegressor(eval(self.kernel["type"])( length_scale=float(self.kernel["length_scale"] ))\
         + WhiteKernel()) ,\
         df=self.df , query=self.query ,x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        return model

    def accuracy( self , model ):
        report_dict = self.acc_calc( model=model , df=self.df , query=self.query ,\
         x_cols=self.x_cols , y_cols=self.y_cols , std=self.std )

        return report_dict

    def dump( self , model ):
        self.model_save( model=model , filename=self.filename ,\
         modelname="gausprocess" , x_cols=self.x_cols , y_cols=self.y_cols )

    def acc_calc( self , model , df , query , x_cols , y_cols , std ):
        if query != "":
            df = df.query( query )

        X = df[x_cols]
        Y = df[y_cols]

        X = self._to_norm( cols=std , X=X )

        X_train , X_test , Y_train , Y_test = train_test_split( X , Y )
        Y_test = np.array( Y_test ).T[0]
        predicted = model.predict( X_test )
        predicted_train = model.predict( X_train )

        N = len( predicted ) + len( predicted_train )

        train_acc = mean_squared_error( np.array(Y_train).T[0] , predicted_train.T[0] )
        test_acc = mean_squared_error( np.array(Y_test), predicted.T[0] )

        return { "N" : N , "train" : train_acc , "test" : test_acc }
