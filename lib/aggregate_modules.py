#!-*-coding:utf-8-*-
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import japanize_matplotlib
import matplotlib.pyplot as plt
from .tools.dict_module import *
from abc import ABCMeta , abstractmethod

__all__ = ["pivot"]

class Aggregate(object,metaclass=ABCMeta):
    @abstractmethod
    def dump( self ):
        """
        出力メソッド
        --------------------------------
        """
        raise NotImplementedError()

class pivot(Aggregate):
    """
    DataFrameからpivot_tableを出力する
    --------------------------------
    df : データフレーム <DataFrame>
        - pandasのdataframeオブジェクトを格納

    cols : 出力するカラムの情報 <dict>
        - cols["agg"] : 集計情報 <array>
        - cols["mode"] : 時間間隔 <string>
    --------------------------------
    """
    def __init__( self , df , cols ):
        self.df = df
        self.agg = cols["agg"]
        self.mode = cols["mode"]

    def dump( self ):
        if self.mode == "month":
            agg = json.loads( self.agg )
            self.df[agg[0]] = pd.to_datetime( self.df[agg[0]] )\
            .dt.strftime("%Y%m%d")
            self.df[agg[0]] = pd.to_datetime( self.df[agg[0]] ).dt.strftime("%Y-%m")
        elif self.mode == "day":
            self.df[agg[0]] = pd.to_datetime( self.df[agg[0]] )

        piv_df = self.df[[agg[0],agg[1],agg[2]]]
        piv_df = piv_df[ piv_df[agg[0]] != "NaT" ]

        pivot = piv_df.pivot_table( index=agg[0] , columns=agg[1] ,\
         values=agg[2] , aggfunc="count" )

        return pivot
