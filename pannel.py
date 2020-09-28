#!-*-coding:utf-8-*-
from .controll import backborn as bbn

__all__ = ["set","preprocess","exec"]

def set( work_dir , mode=2 ):
    """
    初期化
    """
    global ini
    ini = bbn( work_dir=work_dir , mode=mode )
    ini.set()

def preprocess():
    """
    前処理
    """
    ini.csv()
    ini.aggregate()
    ini.describe()

def exec():
    """
    実行
    """
    ini.learn()
    ini.simuration()
