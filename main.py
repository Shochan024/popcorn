#!-*-coding:utf-8-*-
import os
from controll import *

ctl = controller( work_dir = "./" , mode=2 )

ctl.csv()
ctl.aggregate()
ctl.describe()
ctl.learn()
