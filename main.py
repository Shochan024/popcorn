#!-*-coding:utf-8-*-
import os
from controll import *

ctl = backborn( work_dir = "./" , mode=2 )

ctl.csv()
ctl.aggregate()
ctl.describe()
ctl.learn()
ctl.simuration()
