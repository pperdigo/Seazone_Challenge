#Import Libraries
from datetime import date
import re
from unicodedata import category
import numpy as np
import pandas as pd
from pyparsing import col
pd.set_option('display.max_columns', None)
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
#Loading and Cleaning the data
_revenue = pd.read_csv("daily_revenue.csv", encoding='utf8')

_revenue = _revenue.astype({
                        'listing': 'category',
                        'date': 'datetime64[ns]',
                        'creation_date': 'datetime64[ns]'
                        },
                         errors='raise')

#Filter out rows where creation date is after date
revenue = _revenue.query('date>=creation_date | creation_date != creation_date')

_listings = pd.read_csv("listings.csv", encoding='utf8')
listings = _listings

#Splitting category and number of rooms because they will be features in upcoming models
categoria = _listings.loc[:,'Categoria'].str.split('(^[^\d]+)',expand=True, regex=True)[1]
qtde_quartos = _listings.loc[:,'Categoria'].str.split('(^[^\d]+)',expand=True, regex=True)[2].str[0:1]
#I had to turn empty text to NaN and Nan to Zero because casting type from '' to int64 would give an error
qtde_quartos = qtde_quartos.replace('', np.nan, regex = True)
qtde_quartos = qtde_quartos.replace(np.nan, '0', regex = True)

listings['Categoria'] = categoria
listings['Qtde_Quartos'] = qtde_quartos

#Separating HOUSE/APARTMENT into its own column
listings['House/Apartment'] = listings['Categoria']

listings.loc[lambda row : row.loc[:,'House/Apartment'].str.contains('HOU'), 'House/Apartment'] = 'HOUSE'
listings.loc[lambda row : ~row.loc[:,'House/Apartment'].str.contains('HOU'),'House/Apartment'] = 'APARTMENT'

#Next step: Filter out strange values of columns in 'listings' (example: 'Banheiros' entry in 'Banheiros' column). Also decide what to do about listings with undefined number of rooms.

# listings = listings.astype({
#                         'listing': 'category',
#                         'Localizacao': 'category',
#                         'Comissao': 'float64',
#                         'Cama Casal': 'int64',
#                         'Cama Solteiro': 'int64',
#                         'Cama Queen': 'int64',
#                         'Cama King': 'int64',
#                         'Sofa Cama Solteiro': 'int64',
#                         'Travesseiros': 'int64',
#                         'Banheiros': 'int64',
#                         },
#                          errors='ignore')

# #Merge listings and revenue for integrated analysis
# df = pd.merge(_revenue, _listings, how='left', on= 'listing')
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
