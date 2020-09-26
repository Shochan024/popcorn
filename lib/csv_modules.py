#!-*-coding:utf-8-*-
import os
import sys
import json
import numpy as np
import pandas as pd
from abc import ABCMeta , abstractmethod
__all__ = ["merge","where","describe"]
class CSVModule(object,metaclass=ABCMeta):
    @abstractmethod
    def __init__( self , path , vals ):
        raise NotImplementedError()

    @abstractmethod
    def dump( self ):
        raise NotImplementedError()

class merge(CSVModule):
    """
    CSVをmergeする
    --------------------------------
    path : mergeする元のcsvのpath <string>

    vals : 出力するカラムの情報 <dict>
        - vals["on"] : 結合key <string>
        - vals["with"] : mergeするファイル名 <string>※必ずmerge元のファイルと同じフォルダに格納しておく
        - vals["mode"] : inner_join or left_join
    --------------------------------
    """
    def __init__( self , path , vals ):
        self.path = path
        self.vals = vals


    def dump( self ):
        on = self.vals["on"]
        csv_file = self.vals["with"]
        join_mode = self.vals["mode"]
        columns = json.loads( self.vals["columns"] )
        file_path = "{}/{}".format( os.path.dirname( self.path ) , csv_file )
        df_1 = pd.read_csv( self.path )
        df_2 = pd.read_csv( file_path )
        df_merged = df_1.merge( df_2 , on=on , how=join_mode )
        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        merged_csv_name = "{}/merged_{}_{}.csv".format( save_path ,\
         os.path.basename( self.path ).split(".")[0] , csv_file.split(".")[0] )

        if columns != "all":
            df_merged = df_merged[columns]


        return { "csv_name" : merged_csv_name , "dataframe": df_merged}


class where(CSVModule):
    def __init__( self , path , vals ):
        self.path = path
        self.vals = vals

    def dump( self ):
        query = self.vals["query"]
        df = pd.read_csv( self.path )
        datetime_columns_arr = self.__datetime_colmuns( df=df )
        for col in datetime_columns_arr:
            df[ df[ col ] =="0000-00-00" ] = np.nan
            df[col] = pd.to_datetime( df[col] )

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        driped_csv_name = "{}/driped_{}.csv".format( save_path ,\
         os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": driped_csv_name , "dataframe": df.query("{}".format(query)) }


    def __datetime_colmuns( self , df ):
        #本来datetime型であるが、strになってしまっているカラムを取得
        datetime_columns_arr = []
        columns = df.columns
        for col in columns:
            arr = list( df[col].dropna() )
            try:
                if isinstance(arr[0],float) is not True and isinstance(arr[0],int) is not True:
                    datetimes = pd.to_datetime( arr[0] )
                    datetime_columns_arr.append( col )

            except:
                continue

        return datetime_columns_arr


class describe(CSVModule):
    """
    DataFrameから集計結果をを出力する
    --------------------------------
    path : 集計する元のcsvのpath <string>

    vals : 出力するカラムの情報 <dict>
    --------------------------------
    """
    def __init__( self , path , vals ):
        self.path = path
        self.vals = vals

    def dump( self ):
        df = pd.read_csv( self.path )
        cols = self.vals["columns"]
        if cols != "all" and cols !="":
            df = df[ json.loads( cols ) ]

        save_path = os.path.dirname( self.path.replace("originals","statistics") )
        save_path = save_path.replace("shaped","statistics")
        described_csv_name = "{}/describe_{}.csv".format( save_path ,\
         os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": described_csv_name , "dataframe": df.describe() }
