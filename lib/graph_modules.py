#!-*-coding:utf-8-*-
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import japanize_matplotlib
import matplotlib.pyplot as plt
from .tools.dict_modules import *
from abc import ABCMeta , abstractmethod
from lifelines.plotting import plot_lifetimes

__all__ = ["lineplot","boxplot","barplot","notnull","unique"]
__all__ += ["pairplot","heatmap","hist","lifetimes","stackedbar","valrate"]

class Describe(object,metaclass=ABCMeta):
    @abstractmethod
    def dump( self ):
        """
        出力メソッド
        --------------------------------
        """
        raise NotImplementedError()

class lineplot(Describe):
    """
    DataFrameから折れ線グラフを出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["x"] : X軸にプロットする変数
        - cols["y"] : 軸にプロットする変数
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df.fillna(0)
        self.x_cols = json.loads( cols["x"] )
        self.y_cols = json.loads( cols["y"] )

    def dump( self ):
        plt.clf()
        fig = plt.figure(figsize=(6,4))
        df_columns = list( self.df.columns )
        df_columns.remove( self.x_cols[0] )
        fig.subplots_adjust(bottom=0.2)
        for col in df_columns:
            plt.plot( self.df[ self.x_cols[0] ] , self.df[ col ] , label=col )

        plt.xticks(rotation=65)
        plt.grid()
        plt.tick_params(labelsize=10)
        plt.legend()

        return fig


class boxplot(Describe):
    """
    DataFrameから箱髭図を出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["x"] : X軸にプロットする変数
        - cols["y"] : 軸にプロットする変数
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df.fillna(0)
        self.x_cols = json.loads( cols["x"] )
        self.y_cols = json.loads( cols["y"] )

    def dump( self ):
        plt.cla()
        fig = plt.figure()
        for i in range( len( self.y_cols ) ):
            plt.bar( self.df[ self.x_cols[0] ] , self.df[ self.y_cols[i] ] )

        return fig

class barplot(Describe):
    """
    DataFrameから折れ線グラフを出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["x"] : X軸にプロットする変数
        - cols["y"] : 軸にプロットする変数
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df
        self.x_cols = json.loads( cols["x"] )
        self.y_cols = json.loads( cols["y"] )

    def dump( self ):
        fig = plt.figure()
        sns.barplot( x=self.x_cols[0] , y=self.y_cols[0] , data=self.df )

        return fig


class notnull(Describe):
    """
    DataFrameから欠損値棒グラフを出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["border"] : 望ましいサンプル数の割合 <array>
        - cols["x"] : 割合の歩幅 <string>
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df
        self.border = json.loads( cols["border"] )
        self.every = json.loads( cols["x"] )
        self.x_title = "変数名"
        self.y_title = "欠損値ではない割合"

    def dump( self ):
        plt.cla()
        sns.set(font='Yu Gothic')
        sns.set( font=["IPAexGothic"], font_scale=10 / 6 )
        sns.set_palette( "deep" )
        sns.set_context( "paper" , 0.8, { "lines.linewidth" : 1 } )
        notnull = self.df.count()
        n = len( self.df )
        keys = notnull.index
        values = notnull.values
        props = np.round( values/n , decimals=3 )
        not_null = sorted( dict( zip( keys , props ) ).items() ,\
         key=lambda x:x[1] , reverse=True )

        not_null = self.__to_dict( not_null )
        not_null = split_dict( dict=not_null , value_every=self.every )

        fig = self.__plot( not_null=not_null )


        return fig

    def __plot( self , not_null ):
        fig = plt.figure(figsize=(7, 7))
        fig.subplots_adjust( hspace=0.3 , wspace=1.0 )
        fig.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.95)
        for i , key in enumerate( not_null ):
            ax = fig.add_subplot( np.ceil( len( not_null ) / 2 ) , 2 , i+1 )

            x_labels = list( not_null[key].keys() )
            y_values = list( not_null[key].values() )
            x = np.arange( 0 , len( x_labels ) , 1 )
            ax.plot( np.ones( x.shape ) * float( self.border ) , x , \
            linestyle="dotted" , color="black" , label="欲しいサンプル率")

            ax.barh( np.arange( len(x_labels) ) ,y_values , tick_label=x_labels )
            ax.set_xlabel( self.x_title )
            ax.set_ylabel( self.y_title )
            ax.set( xlim=(0,1) )
            ax.set_aspect('auto')

        return fig

    def __to_dict( self , zip_list ):
        """
        sorted(dict)により[("key","value")]と出力されてしまうものをdict型に戻す
        --------------------------------
        input : zipされたlist
        return : dict
        --------------------------------
        """
        dict = {}
        for l in zip_list:
            dict[l[0]] = l[1]

        return dict


class unique(Describe):
    """
    DataFrameからunique件数の棒グラフを出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["x"] : 割合の歩幅 <string>
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df
        self.x_cols = json.loads( cols["x"] )
        self.x_title = "変数名"
        self.y_title = "欠損値ではない割合"

    def dump( self ):
        plt.cla()
        sns.set(font='Yu Gothic')
        sns.set( font=["IPAexGothic"], font_scale=10 / 6 )
        sns.set_palette( "deep" )
        sns.set_context( "paper" , 0.8, { "lines.linewidth" : 1 } )

        fig = plt.figure()
        X = np.array( self.df[self.x_cols] ).astype(np.str)
        plot_dict = {}
        for i in range(X.shape[1]):
            plot_dict[self.x_cols[i]] = np.unique( X[:,i] ).shape[0]

        labels = list( plot_dict.keys() )
        values = list( plot_dict. values() )
        plot_x = np.arange(1,len(labels)+1)
        plt.barh( plot_x , values )
        plt.yticks( plot_x , labels )

        return fig


class pairplot(Describe):
    """
    DataFrameからPairPlotを出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["border"] : 望ましいサンプル数の割合 <array>
        - cols["x"] : 割合の歩幅 <string>
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df
        self.cols = cols

    def dump( self ):
        hue = self.cols["hue"]
        if self.cols["x"] !="" and self.cols["x"] !="all":
            self.df = self.df[ json.loads( self.cols["x"] ) ]

        if hue != "":
            fig = sns.pairplot( self.df , hue=hue , diag_kind='kde' )
        else:
            fig = sns.pairplot( self.df , diag_kind='kde' )

        return fig


class heatmap(Describe):
    """
    DataFrameからHeatmapを出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["border"] : 望ましいサンプル数の割合 <array>
        - cols["x"] : 割合の歩幅 <string>
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df.dropna()
        self.cols = cols

    def dump( self ):
        fig = plt.figure()
        if self.cols["x"] !="" and self.cols["x"] !="all":
            self.df = self.df[ json.loads( self.cols["x"] ) ]

        sns.heatmap( self.df.corr() , annot=True )


        return fig

class stats_describe(Describe):
    """
    DataFrameから基本統計量グラフを出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["border"] : 望ましいサンプル数の割合 <array>
        - cols["x"] : 割合の歩幅 <string>
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df
        self.cols = cols

    def dump( self ):
        pass

class hist(Describe):
    """
    DataFrameからヒストグラムを出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["border"] : 望ましいサンプル数の割合 <array>
        - cols["x"] : 割合の歩幅 <string>
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df
        self.cols = cols

    def dump( self ):
        fig = plt.figure()

        X = json.loads( self.cols["x"] )
        plt.title( "_".join(X) )
        X = np.array( self.df[X].dropna() )
        plt.hist( X )

        return fig


class stackedbar(Describe):
    """
    DataFrameからヒストグラムを出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["border"] : 望ましいサンプル数の割合 <array>
        - cols["x"] : 割合の歩幅 <string>
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df
        self.cols = cols

    def dump( self ):
        fig, axes = plt.subplots(nrows = len(self.cols), ncols = 2, figsize = (18, 20))
        categories = json.loads( self.cols["x"] )
        y = self.df[json.loads( self.cols["y"] )[0]]

        for c in categories:
            freqs = pd.crosstab( self.df[c] , y )
            freqs.plot( ax = axes[ categories.index(c) , 0 ], kind = 'bar', stacked = True )
            axes[ categories.index(c) ][0].set_xticklabels( freqs.index , rotation=45 , size=12 )
            props = freqs.div( freqs.sum(1).astype(float) , axis = 0 )
            props.plot( ax = axes[ categories.index(c) , 1 ], kind = 'bar', stacked = True )
            axes[ categories.index(c) ][1].set_xticklabels( props.index , rotation = 45 , size = 12 )

            fig.tight_layout()

        return fig

class valrate(Describe):
    """
    DataFrameからヒストグラムを出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["border"] : 望ましいサンプル数の割合 <array>
        - cols["x"] : 割合の歩幅 <string>
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df
        self.cols = cols

    def dump( self ):
        num_col = json.loads( self.cols["x"] )
        y = self.df[json.loads( self.cols["y"] )[0]]

        fig, axes = plt.subplots(nrows = len( num_col ), ncols = 2, figsize = (18, 20))

        for c in num_col:
            freqs = pd.crosstab( self.df[ c ] , y )
            freqs.plot( ax = axes[ num_col.index( c ) , 0 ], kind = 'area' )

            props = freqs.div( freqs.sum(1).astype(float) , axis = 0 )
            props.plot( ax = axes[ num_col.index( c ), 1 ], kind = 'area' )

            fig.tight_layout()

        return fig


class lifetimes(Describe):
    """
    DataFrameからヒストグラムを出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["border"] : 望ましいサンプル数の割合 <array>
        - cols["x"] : 割合の歩幅 <string>
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df
        self.cols = cols

    def dump( self ):
        fig , ax = plt.subplots()
        ax = plot_lifetimes( self.df[ json.loads( self.cols["x"] )[0] ]  ,\
         event_observed=self.df[ json.loads( self.cols["y"] )[0] ] )
        ax.set_xlabel( "day" )
        ax.set_ylabel( "survive start and stop" )

        return fig
