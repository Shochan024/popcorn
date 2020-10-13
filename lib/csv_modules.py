#!-*-coding:utf-8-*-
import os
import sys
import json
import numpy as np
import pandas as pd
import category_encoders as ce
from .tools.file_modules import *
from abc import ABCMeta , abstractmethod
__all__ = ["merge","where","categorical","withoutdup","renamecol","replacecol"]

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
        df_1 = self.df
        df_2 = pd.read_csv( csv_file )
        df_merged = df_1.merge( df_2 , on=on , how=join_mode )
        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        merged_csv_name = "merged_{}_{}".format( os.path.basename( self.path ).split(".")[0] , csv_file.split(".")[0].replace("/","_") )

        if columns[0] != "all":
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
        df = self.df
        if len( self.cols ) > 0:
            df = df[ self.cols ]
        datetime_columns_arr = datetime_colmuns( df=df )
        for col in datetime_columns_arr:
            df[ df[ col ] =="0000-00-00" ] = np.nan
            df[col] = pd.to_datetime( df[col] )

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        driped_csv_name = "driped_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": driped_csv_name , "save_path" : save_path , "dataframe": df.query("{}".format(query)) }

class renamecol(CSVModule):
    def __init__( self , path , vals , df ):
        self.path = path
        self.vals = vals
        self.df = df

    def dump( self ):
        df = self.df.rename( columns={ self.vals["before"] : self.vals["after"] } )
        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        renamed_csv_name = "renamed_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": renamed_csv_name , "save_path" : save_path , "dataframe": df }

class replacecol(CSVModule):
    def __init__( self , path , vals , df ):
        self.path = path
        self.vals = vals
        self.df = df

    def dump( self ):
        df = self.df
        col = np.where( np.array(df[self.vals["column"]]) == self.vals["before"] , 1,0 )
        df[self.vals["column"]] = col

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        renamed_csv_name = "renamed_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": renamed_csv_name , "save_path" : save_path , "dataframe": df }


class withoutdup(CSVModule):
    def __init__( self , path , vals , df ):
        self.path = path
        self.vals = vals
        self.df = df

    def dump( self ):
        col = self.vals["column"]
        df = self.df
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
        df = self.df
        list_cols = json.loads( self.vals["columns"] )
        ce_ohe = ce.OrdinalEncoder( cols=list_cols,handle_unknown='impute' )
        df = ce_ohe.fit_transform( df )

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        replaced_csv_name = "replaced_csv_name_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name" : replaced_csv_name , "save_path" : save_path , "dataframe" : df }
