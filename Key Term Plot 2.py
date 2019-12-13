# INSERT YOUR VARIABLES ------------------------------------------------------

input_directory = "C:/Users/Florian Gehrig/Box/02. Live Engagements/BayWa r.e. - Internal Engagement (2019)/03-Rollout/01-Rework Strategic Directives/99-Employee Feedback Survey/00-Data/2019-08-18-Rohdaten von Umfrage XXX._employee_survey (alle Teilnehmer)_WHOLE_FINAL_v4.csv"
output_directory = "C:/Users/Florian Gehrig/Documents"
variables_of_interest = ["Q7 - 1", "Q7 - 2", "Q7 - 3"]


###############################################################################


# 1. LIBRARY LOADING ----------------------------------------------------------

import sys
sys.path.insert(0, 'C:/Users/Florian Gehrig/Documents/GitHub/Analytics-Python')
from textanalytics import *
import pandas as pd


# 2. SETTINGS -----------------------------------------------------------------

pd.set_option('display.max_colwidth', -1) # PRINT INTO CONSOLE
os.chdir(output_directory) #PRINT INTO DOCUMENTS


# 3. DATA LOADING -------------------------------------------------------------

df = pd.read_csv(input_directory, encoding='latin-1')

text_list=[]

for i in variables_of_interest:
            text_list.append(df[i].tolist()) # MERGE ALL DATAFRAMES

text = [text for variable in text_list for text in variable]


# 4. TEXT PREPROCESSING -------------------------------------------------------

# 4.1 TEXT CLEANING ---

text_cleaned = text_cleaner(text)

# 4.2 PHRASE EXTRACTION ---

text_cleaned_ngramed = ngram_extract(text_cleaned,min_freq=4)
#corpus =[["".join(word)] for word in text_cleaned_ngramed if word not in stop_words and len(word)>2]


# 5. PERFORM ANALYSIS   -----------------------------------------------------------

output = key_term_plot(corpus,dims=[3],scaling=1.5, as_interactive = True)



