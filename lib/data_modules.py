#!-*-coding:utf-8-*-
import os
import numpy as np
import pprint as pp
import pandas as pd
from .tools.logger import *

__all__ = ["OutLook"]

class OutLook:
    """
    DataFrameの特徴を表現するClass
    --------------------------------
    df : データフレーム
        - pandasのdataframeオブジェクトを格納

    report : 結果をレポートするか否かの設定
        - Boolean型
    --------------------------------
    """
    def __init__( self , df , report=False ):
        self.df = df
        self.columns = df.columns
        self.row_num = len( df )
        self.report = report


    def info( self , report=False ):
        """
        データの基本情報を可視化するメソッド
        --------------------------------
        input : なし
        return : なし
        --------------------------------
        """
        if report and self.report:
            info = self.df.info()
            return info


    def notnull( self , report=False ):
        """
        データセットの欠損値ではないデータの数をカウントするメソッド
        --------------------------------
        input : なし
        return : dict
        --------------------------------
        """
        not_null = self.df.count()
        props = np.round( not_null.values / len( self.df ) , decimals=3 )
        keys = list( not_null.index )
        not_null = sorted( dict( zip( keys , props ) ).items() , key=lambda x:x[1] , reverse=True )

        self._report( df=not_null , report=report )
        not_null = self._to_dict( zip_list=not_null )

        return not_null

    def merge_aveil( self , joint_key , right_df , report=False ):
        """
        self.dfをleftとして、引数のdfを結合可能かを判定
        --------------------------------
        input : なし
            - joint_key : 結合キー。共通の名前で指定するため、予めrenameしておくように
            - right_df : 結合するデータフレーム
        return : boolean
        --------------------------------
        """

        #結合ロジックはAかつBなので、不要
        if len( list( self.df[joint_key] ) )  <= len( list( right_df[joint_key] ) ):
            return True
        else:
            return False

    def unique( self , col , report=False ):
        """
        対象のカラムのユニーク配列を出力するメソッド
        --------------------------------
        input :
            - col : string カラム名
        return : dict
        --------------------------------
        """
        return np.unique( np.array( list( self.df[col] ) ) )

    def cols( self , report=False ):
        """
        データフレームのカラムをリストで取得するメソッド
        --------------------------------
        input : なし
        return : list
        --------------------------------
        """
        colmuns = self.df.columns()
        self._report( df=colmuns , report=report )

        return columns

    def head( self , report=False ):
        """
        データフレームの上から5行を取得する
        --------------------------------
        input : なし
        return : なし
        --------------------------------
        """
        self._report( df=self.df.head() , report=report )

        return self.df.head()

    def rem_columns( self , req_list , report=False ):
        """
        不要なカラムを削除
        --------------------------------
        input :
            req_list : 必要なリスト
        return : なし
        --------------------------------
        """
        self.df = self.df[ req_list ]
        self.columns = self.df.columns

    def dump( self , filepath , report=False ):
        """
        データフレームをCSVに出力する
        --------------------------------
        input :
            - filepath : 出力先のファイル名
        return : なし
        --------------------------------
        """
        dirname = os.path.dirname( filepath )
        if os.path.exists( dirname ) is not True:
            system( "mkdir {}".format( dirname ) )
            os.makedirs( dirname )
        system( "csv dumping..." )
        self.df.to_csv( filepath )
        system( "merged csv saved in {}".format( dirname ) )


    def _report( self , df , report=False ):
        """
        個別のreport設定がTrueでかつオブジェクトのreport設定がTrueのときのみ
        --------------------------------
        input :
            - df : dataframe
            - report : 結果をprintするか
        --------------------------------
        """
        if report and self.report:
            pp.pprint( df )

    def _to_dict( self , zip_list ):
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
