import numpy as pd
import pandas as pd

revenue = pd.read_csv("daily_revenue.csv", encoding='utf8')
listings = pd.read_csv("listings.csv", encoding='utf8')


df = revenue.merge(listings, on='listing', how='left')
df.to_excel('left_join.xlsx')