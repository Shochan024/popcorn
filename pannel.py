#!-*-coding:utf-8-*-
import os
import sys
sys.path.append( os.path.dirname(__file__) )
from controll import backborn as bbn

__all__ = ["datapop"]

class datapop:
    def __init__( self , work_dir , exec_array=["csv","aggregate","describe","learn","simuration"] , mode=2 ):
        self.ini = bbn( work_dir=work_dir , mode=mode )
        self.exec_array = exec_array


    def process( self ):
        """
        実行
        """
        for exec in self.exec_array:
            eval("self.ini.{}".format(exec))()
