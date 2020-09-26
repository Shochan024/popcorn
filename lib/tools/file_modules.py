#!-*-coding:utf-8-*-
import pandas as pd
import json
__all__ = ["json2dict","datetime_colmuns"]

def json2dict( json_path ):
    json_file = open( json_path , 'r' )
    parse = json.load( json_file )

    return parse

def datetime_colmuns( df ):
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
