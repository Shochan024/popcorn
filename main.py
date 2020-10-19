#!-*-coding:utf-8-*-
import os
import sys
import pannel
exec_array = []
for arg in sys.argv[1:]:
    exec_array.append( arg )
if len( exec_array ) == 0:
    exec_array = ["csv","aggregate","describe","learn","simuration"]

exec = pannel.datapop( work_dir="./" , exec_array=exec_array , mode=2 )
exec.process()
