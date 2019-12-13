from google.cloud import translate
import pandas as pd
import numpy as np
import os

path = "C:/Users/Florian Gehrig/Box/02. Live Engagements/BayWa r.e. - Internal Engagement (2019)/03-Rollout/01-Rework Strategic Directives/99-Employee Feedback Survey/00-Data/BYW105 - Employee Survey-99c79749aa63.json"
df=pd.read_csv("C:/Users/Florian Gehrig/Box/02. Live Engagements/BayWa r.e. - Internal Engagement (2019)/03-Rollout/01-Rework Strategic Directives/99-Employee Feedback Survey/00-Data/2019-08-18-Rohdaten von Umfrage XXX._employee_survey (alle Teilnehmer)_WHOLE_FINAL_v4.csv")

def translating(path, df):
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
    
    translate_client = translate.Client()
    
    df = df
    translates = []
    
    for x, column in enumerate(df.columns[30:]):
        if df[column].dtype.kind != 'O':
            print(column,"Column not needed")
        else:
            to_trans = df[column].tolist()
            for i in range(len(to_trans)):
                if isinstance(to_trans[i], float) == True or str(to_trans[i]) in ["-99","-66","0"] or to_trans[i] in ["NaN",np.NaN] or list(translate_client.detect_language(to_trans[i]).values())[1] == "en":
                    df[column][i]=to_trans[i]
                    print("NT",to_trans[i])
                else:
                    try:
                        translation = translate_client.translate(to_trans[i], target_language='en')
                        df[column][i] = translation['translatedText']
                        print("T",to_trans[i])
                        translates.append(to_trans[i])
                        
                    except:
                        df[column][i]=to_trans[i]
                        print("NT",to_trans[i])    
                        
    return df, translates
        
df_translated3, translates = translating(path, df_translated)


df_translated3.to_csv("C:/Users/Florian Gehrig/Box/02. Live Engagements/BayWa r.e. - Internal Engagement (2019)/03-Rollout/01-Rework Strategic Directives/99-Employee Feedback Survey/00-Data/2019-08-18-Rohdaten von Umfrage XXX._employee_survey (alle Teilnehmer)_WHOLE_FINAL_v6.csv")