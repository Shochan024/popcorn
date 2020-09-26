#!-*-coding:utf-8-*-
import os
from controll import *

ctl = controller( data_path = "./"\
, setting_path="./" , mode=2 )

ctl.csv()
ctl.aggregate()
ctl.describe()
ctl.predict()
