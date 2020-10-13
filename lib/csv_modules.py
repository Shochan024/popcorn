#!-*-coding:utf-8-*-
import os
import sys
import json
import numpy as np
import pandas as pd
import category_encoders as ce
from .tools.file_modules import *
from abc import ABCMeta , abstractmethod
__all__ = ["merge","where","categorical","valuecounts","withoutdup"]
class CSVModule(object,metaclass=ABCMeta):
    @abstractmethod
    def __init__( self , path , vals , df ):
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
    def __init__( self , path , vals , df ):
        self.path = path
        self.vals = vals
        self.df = df


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
        merged_csv_name = "merged_{}_{}".format( os.path.basename( self.path ).split(".")[0] , csv_file.split(".")[0] )

        if columns != "all":
            df_merged = df_merged[columns]


        return { "csv_name" : merged_csv_name , "save_path" : save_path , "dataframe": df_merged}


class where(CSVModule):
    def __init__( self , path , vals , df ):
        self.path = path
        self.vals = vals
        self.df = df
        self.cols = json.loads( self.vals["columns"] )

    def dump( self ):
        query = self.vals["query"]
        df = pd.read_csv( self.path )
        if len( self.cols ) > 0:
            df = df[ self.cols ]
        datetime_columns_arr = datetime_colmuns( df=df )
        for col in datetime_columns_arr:
            df[ df[ col ] =="0000-00-00" ] = np.nan
            df[col] = pd.to_datetime( df[col] )

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        driped_csv_name = "driped_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": driped_csv_name , "save_path" : save_path , "dataframe": df.query("{}".format(query)) }

class valuecounts(CSVModule):
    def __init__( self , path , vals , df ):
        self.path = path
        self.vals = vals
        self.df = df

    def dump( self ):
        col = self.vals["column"]
        df = pd.read_csv( self.path )
        df = df[ col ]
        save_path = os.path.dirname( self.path.replace("originals","statistics") )
        save_path = save_path.replace("shaped","statistics")
        value_count_csv_name = "value_count_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": value_count_csv_name , "save_path":save_path , "dataframe": df.value_counts() }

class withoutdup(CSVModule):
    def __init__( self , path , vals , df ):
        self.path = path
        self.vals = vals
        self.df = df

    def dump( self ):
        col = self.vals["column"]
        df = pd.read_csv( self.path )
        df = df[ ~df[ col ].duplicated() ]

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        save_path = save_path.replace("shaped","shaped")
        value_count_csv_name = "withoutdup_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": value_count_csv_name , "save_path" : save_path , "dataframe": df }


class categorical(CSVModule):
    """
    DataFrameの特定の列を置換する
    --------------------------------
    path : 集計する元のcsvのpath <string>

    vals : 出力するカラムの情報 <dict>
    --------------------------------
    """
    def __init__( self , path , vals , df ):
        self.path = path
        self.vals = vals
        self.df = df

    def dump( self ):
        df = pd.read_csv( self.path )
        list_cols = json.loads( self.vals["columns"] )
        ce_ohe = ce.OrdinalEncoder( cols=list_cols,handle_unknown='impute' )
        df = ce_ohe.fit_transform( df )

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        replaced_csv_name = "replaced_csv_name_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name" : replaced_csv_name , "save_path" : save_path , "dataframe" : df }
