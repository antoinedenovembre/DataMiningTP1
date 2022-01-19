# Import libraries
"""Packages"""
from introduction.intro import *

""" Main """
import numpy as np
import pandas as pd
import os, time
import pickle, gzip

'''Data Viz'''
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

df_train = pd.read_csv('../Data/weatherAUS.csv', sep=",")
main_introduction(df_train)
