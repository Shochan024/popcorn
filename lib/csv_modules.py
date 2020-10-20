#!-*-coding:utf-8-*-
import os
import sys
import json
import numpy as np
import pandas as pd
import category_encoders as ce
from .tools.file_modules import *
from .tools.logger import *
from abc import ABCMeta , abstractmethod
__all__ = ["merge","where","categorical","withoutdup","renamecol","logcol"]
__all__ += ["replacecol","fillna","crossterm","dup","groupbycount","timediff"]

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
        df = df_1.merge( df_2 , on=on , how=join_mode )
        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        merged_csv_name = "merged_{}_{}".format( os.path.basename( self.path ).split(".")[0] , csv_file.split(".")[0].replace("/","_") )

        if columns[0] != "all":
            df = df[columns]

        df = df[~df.duplicated()]


        return { "csv_name" : merged_csv_name , "save_path" : save_path , "dataframe": df}


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

class crossterm(CSVModule):
    def __init__( self , path , vals , df ):
        self.path = path
        self.vals = vals
        self.df = df

    def dump( self ):
        df = self.df
        col1 = self.vals["col1"]
        col2 = self.vals["col2"]
        df["{}*{}".format(col1,col2)] = df[col1] * df[col2]

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        crossterm_csv_name = "crossterm_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": crossterm_csv_name , "save_path" : save_path , "dataframe": df }

class logcol(CSVModule):
    def __init__( self , path , vals , df ):
        self.path = path
        self.vals = vals
        self.df = df

    def dump( self ):
        df = self.df
        cols = json.loads( self.vals["columns"] )
        for col in cols:
            df = df[ df[col] > 0 ]
            df["log_{}".format( col )] = np.round( np.log( np.array( df[col] ) ) , 5 )

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        log_csv_name = "log_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": log_csv_name , "save_path" : save_path , "dataframe": df }

class dup(CSVModule):
    def __init__( self , path , vals , df ):
        self.path = path
        self.vals = vals
        self.df = df

    def dump( self ):
        df = self.df
        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        dup_csv_name = "{}_{}".format( self.vals["filename"] , os.path.basename( self.path ).split(".")[0] )
        df.to_csv( "{}/{}.csv".format( save_path , dup_csv_name ) )
        message( "save figure in {}/{}".format( save_path , dup_csv_name ) )

        return { "csv_name": False , "save_path" : save_path , "dataframe": df }

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
        if self.vals["before"] == "notnull":
            col = np.array( df[self.vals["column"]].fillna( 0 ) )
            col = np.where( col != 0 , 1 , 0 )
        else:
            col = np.where( np.array(df[self.vals["column"]]) == self.vals["before"] , 1,0 )

        df["{}_replaced".format(self.vals["column"])] = col

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        renamed_csv_name = "renamed_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": renamed_csv_name , "save_path" : save_path , "dataframe": df }

class groupbycount(CSVModule):
    def __init__( self , path , vals , df ):
        self.path = path
        self.cols = json.loads( vals["cols"] )
        self.df = df
        self.groupby = vals["groupby"]

    def dump( self ):
        col = self.cols
        df = self.df
        df = df[self.cols].groupby( self.groupby ).count()

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        groupbycount_csv_name = "groupbycount_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": groupbycount_csv_name , "save_path" : save_path , "dataframe": df }

class fillna(CSVModule):
    def __init__( self , path , vals , df ):
        self.path = path
        self.vals = vals
        self.df = df

    def dump( self ):
        df = self.df
        for col in json.loads( self.vals["columns"] ):
            if self.vals["how"] == "mean":
                df[col] = df[col].fillna( df[col].mean() )

            elif self.vals["how"] == "zero":
                df[col] = df[col].fillna( 0 )

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        fillna_csv_name = "fillna_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name": fillna_csv_name  , "save_path" : save_path , "dataframe": df }


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

        if self.vals["mode"] == "onehot":
            ce_ohe = ce.OneHotEncoder( cols=list_cols,handle_unknown='impute' )
        else:
            ce_ohe = ce.OrdinalEncoder( cols=list_cols,handle_unknown='impute' )
        df = ce_ohe.fit_transform( df )

        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        replaced_csv_name = "replaced_csv_name_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name" : replaced_csv_name , "save_path" : save_path , "dataframe" : df }

class timediff(CSVModule):
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
        left_item = pd.to_datetime( df[self.vals["left_time"]] )
        right_item = pd.to_datetime( df[self.vals["right_time"]] )
        diff = left_item - right_item
        diff = np.abs( diff.dt.days )
        df["{}_{}_diff".format( self.vals["left_time"] , self.vals["right_time"] )] = diff


        save_path = os.path.dirname( self.path.replace("originals","shaped") )
        replaced_csv_name = "replaced_csv_name_{}".format( os.path.basename( self.path ).split(".")[0] )

        return { "csv_name" : replaced_csv_name , "save_path" : save_path , "dataframe" : df }
