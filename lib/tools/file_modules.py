#!-*-coding:utf-8-*-
import json
__all__ = ["json2dict"]

def json2dict( json_path ):
    json_file = open( json_path , 'r' )
    parse = json.load( json_file )

    return parse
