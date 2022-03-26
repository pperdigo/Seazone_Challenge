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
def CleanAndMerge(_revenue, _listings):
    # # Loading and Cleaning the data
    # Revenue
    _revenue = _revenue.astype({
                            'listing': 'category',
                            'date': 'datetime64[ns]',
                            'creation_date': 'datetime64[ns]'
                            },
                            errors='raise')

    #Filter out rows where creation date is after date or creation date is null
    revenue = _revenue.query('date>=creation_date | creation_date != creation_date')
    revenue_final = revenue.drop_duplicates()
    # Listings
    listings = _listings

    #Splitting category and number of rooms because they will be features in upcoming models
    categoria = _listings.loc[:,'Categoria'].str.split('(^[^\d]+)',expand=True, regex=True)[1]
    qtde_quartos = _listings.loc[:,'Categoria'].str.split('(^[^\d]+)',expand=True, regex=True)[2].str[0:1]

    qtde_quartos = qtde_quartos.replace('', np.nan, regex = True).astype('Int64')

    listings['Categoria'] = categoria
    listings['Qtde_Quartos'] = qtde_quartos

    #Separating House_or_Apartment into its own column
    listings['House_or_Apartment'] = listings['Categoria']

    listings.loc[lambda row : row.loc[:,'House_or_Apartment'].str.contains('HOU'), 'House_or_Apartment'] = 'HOUSE'
    listings.loc[lambda row : ~row.loc[:,'House_or_Apartment'].str.contains('HOU'),'House_or_Apartment'] = 'APARTMENT'

    # Dropping rows where 'Tipo' doesn't match 'HOU' prefix
    listingsNotMatchingHouse = listings.query("House_or_Apartment == 'APARTMENT' & Tipo == 'Casa'")['listing']
    listings.drop(listingsNotMatchingHouse.index, inplace=True)
    listingsNotMatchingApartment = listings.query("House_or_Apartment == 'HOUSE' & Tipo != 'Casa'")['listing']
    listings.drop(listingsNotMatchingApartment.index, inplace=True)
    listings.drop(['House_or_Apartment'], axis=1, inplace=True)

    # Remove 'HOU' prefix
    listings['Categoria'] = listings['Categoria'].str.removeprefix('HOU')

    # Correct 'TOPM' typo
    listings['Categoria'] = listings['Categoria'].str.replace('TOPM', 'TOP')

    # Remove specific entries that don't match the column context
    columnsToClean = ['Cama Casal', 'Cama Solteiro', 'Cama Queen', 'Cama King', 'Sofa Cama Solteiro', 'Banheiros', 'Capacidade']
    valuesToBeRemoved = ['Quantidade de Camas Casal', 'Quantidade de Camas Solteiro', 'Quantidade de Camas Queen',
                        'Quantidade de Camas King', 'Quantidade de Sofás Cama Solteiro', 'Banheiros', 'Capacidade']
    for coluna in columnsToClean:
        indexToDrop = listings[listings[coluna].isin(valuesToBeRemoved)].index
        listings.drop(indexToDrop,inplace=True)

    # Cast correct dtypes to columns
    for column in listings.columns:
        if column != 'Endereco' and listings[column].dtype == 'object':
            listings[column] = listings[column].str.replace(',', '.')
    listings = listings.astype({
                            'listing': 'category',
                            'Localizacao': 'category',
                            'Comissao': 'float64',
                            'Cama Casal': 'float64',
                            'Cama Solteiro': 'float64',
                            'Cama Queen': 'float64',
                            'Cama King': 'float64',
                            'Sofa Cama Solteiro': 'float64',
                            'Travesseiros': 'float64',
                            'Banheiros': 'float64',
                            'Taxa de Limpeza': 'float64',
                            'Capacidade': 'float64'
                            },
                            errors='raise')

    listings_final = listings.drop_duplicates()

    #Merge listings and revenue for integrated analysis
    df = pd.merge(revenue_final, listings_final, how='left', on= 'listing')
    return df
