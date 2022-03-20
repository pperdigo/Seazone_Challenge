# Data Cleaning Log
## In this markdown, I will track the decisions I made regarding data cleaning. Level 3 headers will point out the challenges and the text below them will explain my thought process to deal with them.

### Missing values about number of rooms.
All entries were supposed to have the number of rooms as a suffix to their category, but 154 of them didn't have it. I made a decision to not cut these listings from my analysis because they represented a large portion of our population, and so I left their NumberOfRooms field as a NaN value, so they can be better dealt with in the future when working with a model.

### 'HOU' suffix not matching 'Tipo' column
Entries 319, 327, 411 and 429 were tagged as Houses in the 'Tipo' column, but did not have the 'HOU' prefix in their 'Categoria' column. Given that the information provided is conflicting and I do not have the resources to confirm which column holds true and that these listings only appear in 2696 out of 288971 rows in the daily_revenue table, I will discard these 4 entries from my analysis.