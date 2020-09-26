#!-*-coding:utf-8-*-
from .controll import controller as ctl

__all__ = ["set","preprocess"]

def set( data_path , setting_path , mode=2 ):
    global ini
    ini = ctl( data_path=data_path , setting_path=setting_path , mode=mode )
    ini.set()
    return False

def preprocess():
    ini.csv()
    ini.aggregate()
    ini.describe()
