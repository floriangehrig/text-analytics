# INSERT YOUR VARIABLES ------------------------------------------------------

project_name="VDI - Mitgliederwuensche"

input_directory = "C:/Users/Florian Gehrig/Documents/VDI - Mitgliederwünsche_v2.csv"
output_directory = "C:/Users/Florian Gehrig/Documents"

variables_of_interest = ["Statement"]
groupings=["Group"]
cols=["Statement"]
n_cols=[13]
font_directory = "C:/Users/Florian Gehrig/Downloads/arialbd.ttf"


###############################################################################

# 1. LIBRARY LOADING ----------------------------------------------------------

# 1.1 DATA MANIPULATION ---

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, (os.path.expanduser(os.getenv('USERPROFILE')).replace('\\','/') +'/Documents/GitHub/Analytics-Python/Text Analytics'))


# 1.2  TEXT ANALYTICS ---

from nltk.corpus import stopwords
from textanalytics import *


# 2. SETTINGS -----------------------------------------------------------------

os.chdir(output_directory)
pd.set_option('display.max_columns', 1000)
np.set_printoptions(threshold=sys.maxsize)

stop_words = set(stopwords.words('english'))

def byw_grey(word, *args, **kwargs):
    return "#4c4c4c"


# 3. DATA LOADING -------------------------------------------------------------

df = pd.read_csv(input_directory, encoding = "ISO-8859-1")


# CROSSTABS -------------------------------------------------------------------

theme_crosstabs(df, cols, n_cols, project_name, groupings)


# THEME CO-OCCURENCE MATRIX ---------------------------------------------------

theme_correlation(df,project_name,cols, n_cols)


# TOPIC-BASED WORDCLOUD CREATION ----------------------------------------------

theme_based_wordcloud(df,cols,n_cols,groupings,font_directory)


# GROUPING-BASED WORDCLOUD CREATION --------------------------------------------

group_based_wordcloud(df, cols, n_cols, groupings, font_directory, color_function=byw_grey(), scaling=1)
