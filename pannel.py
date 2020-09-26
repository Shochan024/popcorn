#!-*-coding:utf-8-*-
from .controll import controller as ctl

__all__ = ["set","preprocess"]

def set( work_dir , mode=2 ):
    global ini
    ini = ctl( work_dir=work_dir , mode=mode )
    ini.set()
    return False

def preprocess():
    ini.csv()
    ini.aggregate()
    ini.describe()
    ini.learn()
