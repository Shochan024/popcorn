#!-*-coding:utf-8-*-
import os
from controll import *

"""
import pandas as pd

df = pd.read_csv("datas/shaped/driped_2_action_history.csv")
print( len( df["顧客ID_customer"] ) )
print( len( df["顧客ID_customer"][df["顧客ID_customer"].duplicated()] ) )
"""

ctl = backborn( work_dir = "./" , mode=2 )

ctl.csv()
ctl.aggregate()
ctl.describe()
ctl.learn()
ctl.simuration()
