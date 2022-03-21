#Import Libraries
from datetime import date
import re
from unicodedata import category
import numpy as np
import pandas as pd
from pyparsing import col
pd.set_option('display.max_columns', None)

#Import previous data cleaning function
from data_cleaning import CleanAndMerge

#Clean and Merge data
_revenue = pd.read_csv('daily_revenue.csv')
_listings = pd.read_csv('listings.csv')
df = CleanAndMerge(_revenue, _listings)
print (df)

