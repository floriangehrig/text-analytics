# -*- coding: utf-8 -*-
"""

FUNCTIONS FOR AUTOMATIC TEXT MINING

TO-DOS:

    - Add Word Type Filter for Wordclouds (e.g. Adjectives, Nouns only)
    - Add Stopword Filters for different languages (+ option to add custom stopwords)

"""
# 1. LIBRARY LOADING ----------------------------------------------------------


# 1.1 DATA MANIPULATION ---

import pandas as pd
import numpy as np
from dfply import *
from collections import Counter
from datetime import datetime
from google.cloud import translate
import os


# 1.2 TEXT MINING ---

#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('words')
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.collocations import *
from nltk.stem import WordNetLemmatizer 
from nltk.metrics.association import QuadgramAssocMeasures
from textblob import TextBlob
import spacy
import regex
import re
from nltk.corpus import stopwords
#import gensim
from gensim.models import Word2Vec


# 1.3 CLUSTERING & DIMENSION REDUCTION ---

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.decomposition import PCA
#from sklearn.metrics import silhouette_score as sc
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import ShuffleSplit


# 1.4 VISUALIZATION ---

import plotly
import plotly.express as px
from plotly.offline import plot
import plotly.figure_factory as ff
from wordcloud import WordCloud



# 2. SETTINGS ----------------------------------------------------------

stop_words = list(set(stopwords.words('english')))
stop_words.extend(['nan','NaN',"/", "people","family","test","no","the"])

DEFAULT_PLOTLY_COLORS = [
    "#636efa",
    "#EF553B",
    "#00cc96",
    "#ab63fa",
    "#FFA15A",
    "#19d3f3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
    "#636efa",
    "#EF553B",
    "#00cc96",
    "#ab63fa",
    "#FFA15A",
    "#19d3f3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]

abbr_dict={
    "what's":"what is",
    "what're":"what are",
    "who's":"who is",
    "who're":"who are",
    "where's":"where is",
    "where're":"where are",
    "when's":"when is",
    "when're":"when are",
    "how's":"how is",
    "how're":"how are",

    "i'm":"i am",
    "we're":"we are",
    "you're":"you are",
    "they're":"they are",
    "it's":"it is",
    "he's":"he is",
    "she's":"she is",
    "that's":"that is",
    "there's":"there is",
    "there're":"there are",

    "i've":"i have",
    "we've":"we have",
    "you've":"you have",
    "they've":"they have",
    "who've":"who have",
    "would've":"would have",
    "not've":"not have",

    "i'll":"i will",
    "we'll":"we will",
    "you'll":"you will",
    "he'll":"he will",
    "she'll":"she will",
    "it'll":"it will",
    "they'll":"they will",

    "isn't":"is not",
    "wasn't":"was not",
    "aren't":"are not",
    "weren't":"were not",
    "can't":"can not",
    "couldn't":"could not",
    "don't":"do not",
    "didn't":"did not",
    "shouldn't":"should not",
    "wouldn't":"would not",
    "doesn't":"does not",
    "haven't":"have not",
    "hasn't":"has not",
    "hadn't":"had not",
    "won't":"will not",
    '\s+':' ', # replace multi space with one single space
}


# 3. FUNCTIONS ----------------------------------------------------------------

def pos_words (sentence, tokens, ptag):
    
    word_dfs = []
    for token in tokens:
        sentences = [sent for sent in sentence.sents if token in sent.string]     
        pwrds = []
        for sent in sentences:
            for word in sent:
                if token in word.string: 
                       pwrds.extend([child.string.strip() for child in word.children if child.pos_ == ptag])
        counts = Counter(pwrds).most_common(1000)
        counts_df=pd.DataFrame(counts)
        counts_df["token"]=len(counts)*[token]
        word_dfs.append(counts_df)
    
    return pd.concat(word_dfs, ignore_index=True)



def ngram_extract(corpus, methods = ["Quadgram","Trigram","Bigram"], min_freq=3):
    
    output = []
    
    for method in methods:
        
        if method == "Bigram":
            
            ngram_measures = nltk.collocations.BigramAssocMeasures()
            ngramFinder = nltk.collocations.BigramCollocationFinder.from_words(corpus)
        
        elif method == "Trigram":
            
            ngram_measures = nltk.collocations.TrigramAssocMeasures()
            ngramFinder = nltk.collocations.TrigramCollocationFinder.from_words(corpus)
            
        elif method == "Quadgram":
            
            ngram_measures = QuadgramAssocMeasures()
            ngramFinder = nltk.collocations.QuadgramCollocationFinder.from_words(corpus)
            
        ngramFinder.apply_freq_filter(min_freq)
        #ngramFinder.apply_word_filter(lambda w: w.lower() in stop_words)
        
        ngram_metrics=pd.DataFrame()
        
        for metric in ["pmi","raw_freq","likelihood_ratio","chi_sq","student_t","jaccard","poisson_stirling"]:
        
            metric_table = pd.DataFrame(list(ngramFinder.score_ngrams(getattr(ngram_measures,metric))), columns=['ngram',metric]).sort_values(by="ngram", ascending=False)
            
            if  ngram_metrics.empty:
                
                ngram_metrics = metric_table
            else:
                ngram_metrics.insert(1,metric,metric_table[metric],True)
        
        if method == "Bigram":
            ngram_metrics = ngram_metrics[ngram_metrics.ngram.map(lambda x: rightTypes(x))]
        elif method == "Trigram":
            ngram_metrics = ngram_metrics[ngram_metrics.ngram.map(lambda x: rightTypesTri(x))]
        elif method == "Quadgram":
            ngram_metrics = ngram_metrics[ngram_metrics.ngram.map(lambda x: rightTypesQuad(x))]
        
        print('!!!!!',ngram_metrics)
        
        if len(ngram_metrics.index) != 0:
            ngram_ranks=pd.DataFrame(ngram_metrics["ngram"])
        
            for column in ngram_metrics.columns[1:]:
                ngram_ranks[column]=ngram_metrics[column].rank(ascending=0)
            
            ngram_ranks['mean'] = ngram_ranks.mean(axis=1)
            final_ngrams = ngram_ranks[["ngram",'mean']].sort_values('mean')
        
            lookup = final_ngrams["ngram"].tolist()
            
            print("\nThese are the extracted "+method+"s: ", lookup)
            
            if method == "Bigram":
                idx = 0
                while idx < (len(corpus)-1):
                    output.append(corpus[idx])
                    if (corpus[idx], corpus[idx+1]) in lookup:
                        output[-1] += str("_"+corpus[idx+1])     
                        idx += 2
                        
                    else:       
                        idx += 1
            
            elif method == "Trigram":
                idx = 0
                while idx < (len(corpus)-2):
                    output.append(corpus[idx])
                    if (corpus[idx], corpus[idx+1], corpus[idx+2]) in lookup:
                        output[-1] += str("_"+corpus[idx+1]+"_"+corpus[idx+2])  
                        idx += 3
                        
                    else:       
                        idx += 1
    
            elif method == "Quadgram":
                idx = 0
                while idx < (len(corpus)-3):
                    output.append(corpus[idx])
                    if (corpus[idx], corpus[idx+1], corpus[idx+2], corpus[idx+3]) in lookup:
                        output[-1] += str("_"+corpus[idx+1]+"_"+corpus[idx+2]+corpus[idx+3])  
                        idx += 4
                        
                    else:       
                        idx += 1
            
            return output
                        
        else:
            return corpus

                        
    #print("Extracted Keywords:")
    #print([string for string in output if '_' in string].unique()) # PRINT ALL N-GRAMS IN CORPUS
    


def text_cleaner(text,remove_nas=False, remove_numbers=False, remove_spacings = False, replace_abbrevations = False, lemmatize = False, grammar_correct = False, remove_social = False,):

    if remove_nas == True: # REMOVE ALL MISSING VALUES

        text = [string for string in text if string not in ["NaN","nan",np.NaN]]
        text = [string for string in text if str(string).lower() not in ['-66','-99','nan']]

    if remove_numbers == True: ## REMOVE ALL NUMBERS

        text=[string for string in text if isinstance(string, float) != True]
        text = [regex.sub(r'[0-9]+', '', string) for string in text]

    #p = re.compile(r"(\b[-#'\.]\b)|[\W]")
    #text = [p.sub(lambda m: (m.group(1) if m.group(1) else " "), string) for string in text]
    
    if remove_spacings == True:  # LOWER AND REMOVE DOUBLE SPACINGS

        text = [regex.sub(r' +', ' ', str(string)).lower() for string in text]
    
    if replace_abbrevations == True:

        text = [' '.join([abbr_dict.get(i, i) for i in string.split()]) for string in text] #REPLACE ABBREVATIONS
    

    text = " ".join([string.lower() for string in text])

    if lemmatize == True:
          
        nlp = spacy.load("en_core_web_sm")
        text = nlp(text)
        text = [token.lemma_ for token in text if not token.is_stop]
        text=list(map(lambda i: i.lower(),text))
        
    if grammar_correct == True: # GRAMMAR CORRECTION (WARNING - NOT TESTED YET)
        
            text = [TextBlob(string).correct() for string in text]
                            
    if remove_social == True:
        
        text = re.sub(r"http\S+", "", text) # To remove web links(http or https) from the tweet text

        text = re.sub(r"#\S+", "", text) # To remove hashtags (trend) from the tweet text

        text = re.sub(r"@\S+", "", text) # To remove user tags from tweet text

        text = re.sub(r"RT", "", text) # To remove re-tweet "RT"

        text = text.replace('\\n','') # To remove new line character if any


    #corpus = ' '.join([word for line in text for word in line.split()])
    corpus = text.split()
    corpus = [string for string in corpus]
    
    return corpus


def rightTypes(ngram): # Filter Bigrams
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in stop_words or word.isspace():
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False


def rightTypesTri(ngram): # Filter Trigrams
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in stop_words or word.isspace():
            return False
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False


def rightTypesQuad(ngram): # Filter Quadgrams
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in stop_words or word.isspace():
            return False
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    fourth_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[3][1] in fourth_type:
        return True
    else:
        return False

# -----------------------------------------------------------------------------


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results



def key_term_plot(df, variables_of_interest, clust_alg="kmeans",title="XXX", client_name = "Text", min_count=3, scaling=2, dims=[2], state=42, as_picture = True, as_interactive=False, custom_tsne_param_grid=dict(),custom_cluster_param_grid=dict() ):
    """
    Key Term Visualization in 2D/3D Space
    
    Description
    ----------
    TBD

    Parameters
    ----------
    corpus : str
        Text Source
    clust_alg : ["kmeans","dbscan"]
        Type of Algorithm to cluster Key Terms
    dims : int [2,3]
        Number of Plotting Dimensions. 
        Can be a list to iterate over.
    client_name : str
        Type of Object which the text is about.
    min_count : int
        Minimum Frequency of a term to be considered for Visualization.
    state : int
        Random State Number for Reproducability
    scaling : int
        Size Scaling of rendered pictures.
    as_picture : bool
        If TRUE, plot will be exported as picture.
    as_interactive: bool
        If TRUE, plit will be exported as interactive HTML file.
    custom_tsne_param_grid : dict
        Custom Parameters for TSNE to iterate over.
        If empty, default parameters will be considered for iteration.
    custom_cluster_param_grid : dict
        Custom Parameters for Clustering to iterate over.
        If empty, default parameters will be considered for iteration.

    Returns
    -------
    int
        Description of return value

    

    Further Features to Consider:
    -----------------------------    
    - Select different term category filters (e.g. Adjectives and Nouns only)
    - Fuzzy Word Matching for merging Similar Adjectives Adverbs
    - PCA opposed to TSNE?   

    """

    text_list=[]

    for i in variables_of_interest:
                text_list.append(df[i].tolist()) # MERGE ALL DATAFRAMES
    
    text = [text for variable in text_list for text in variable]

    text_cleaned = text_cleaner(text)
    text_cleaned_ngramed = ngram_extract(text_cleaned,min_freq=4)

    corpus =[["".join(word)] for word in text_cleaned_ngramed if len(word)>2]

    adjectives = []

    for category in ["a","s","r"]:
        for x in wn.all_synsets(category):
            adjectives.append(x.name().split('.', 1)[0])
    
    model = Word2Vec(corpus,min_count=min_count)

    words = []
    embeddings = []
    freqs = []
    
    for word in model.wv.vocab:
        if word in adjectives:
            try:
                embeddings.append(model[word])
            except KeyError:
                embeddings.append(model["none"])
            words.append(word)
            freqs.append(model.wv.vocab[word].count)
    
    for n_dimensions in dims:
        
        def make_generator(parameters):
            if not parameters:
                yield dict()
            else:
                key_to_iterate = list(parameters.keys())[0]
                next_round_parameters = {p : parameters[p]
                            for p in parameters if p != key_to_iterate}
                for val in parameters[key_to_iterate]:
                    for pars in make_generator(next_round_parameters):
                        temp_res = pars
                        temp_res[key_to_iterate] = val
                        yield temp_res
        
        # TSNE ----------------------------------------------------------------
        
        fixed_tsne_params = {"n_components": n_dimensions, "random_state": state , "n_iter": 1000} 
                
        if not custom_tsne_param_grid:
            tsne_param_grid = {"perplexity": range(5,50,10), "learning_rate": range(100,1000,100)}

        for tsne_params in make_generator(tsne_param_grid):
            final_tsne_params = {**tsne_params, **fixed_tsne_params}
            
            tsne = TSNE(**final_tsne_params)
            tsne_results = tsne.fit_transform(embeddings)
            
            # CLUSTERING ------------------------------------------------------
            
            fixed_cluster_params = {"n_jobs": -1 , "random_state": state , "max_iter": 1000} 
            
            if not custom_cluster_param_grid:
                cluster_param_grid = {"n_clusters": range(3,8)}
        
            for cluster_params in make_generator(cluster_param_grid):
                final_cluster_params = {**cluster_params, **fixed_cluster_params}
                
                df=pd.DataFrame()
                df['Dimension 1'] = tsne_results[:,0]
                df['Dimension 2'] = tsne_results[:,1]
                if n_dimensions == 3:
                    df['Dimension 3'] = tsne_results[:,2]
        
                db = KMeans(**final_cluster_params).fit(df)
        
                df['Frequency'] = freqs
                df['Word'] = words
                df['Value Cluster'] = db.labels_
                df = df.sort_values('Value Cluster')
                df['Value Cluster'] = df['Value Cluster'].astype('category')
                
                
                # PLOTTING ----------------------------------------------------
                
                if n_dimensions == 2: # 2-DIMESIONAL
                    
                    fig = px.scatter(data_frame=df,
                                 x='Dimension 1',
                                 y='Dimension 2',
                                 size='Frequency',
                                 text="Word",
                                 color = 'Value Cluster',
                                 size_max=75,
                                 opacity=0.6,
                                 width=900*scaling,
                                 height=600*scaling,
                                 hover_name='Word')
                    
                    fig.update_traces(textposition='middle center', textfont=dict(size=8))
                    
                    for m, s in enumerate(np.unique(df['Value Cluster']).tolist()):
                        
                        fig.data[m].name = ("Value Cluster "+str(m+1))
                        
                        fig.layout.shapes = fig.layout.shapes + ({
                                    "type":"circle",
                                    "layer":"below",
                                    "xref":"x",
                                    "yref":"y",
                                    "x0":min(df[df['Value Cluster'] == s]['Dimension 1']),
                                    "y0":min(df[df['Value Cluster'] == s]['Dimension 2']),
                                    "x1":max(df[df['Value Cluster'] == s]['Dimension 1']),
                                    "y1":max(df[df['Value Cluster'] == s]['Dimension 2']),
                                    "opacity":0.1,
                                    "fillcolor": DEFAULT_PLOTLY_COLORS[int(s)],
                                    "line_color":DEFAULT_PLOTLY_COLORS[int(s)],
                                },)
            
                    
            
                elif n_dimensions == 3: # 3-DIMENSIONAL 
                    
                        fig = px.scatter_3d(data_frame=df,
                             x='Dimension 1',
                             y='Dimension 2',
                             z='Dimension 3',
                             size='Frequency',
                             text="Word",
                             color = 'Value Cluster',
                             size_max=75,
                             opacity=0.6,
                             width=900*scaling,
                             height=600*scaling,
                             hover_name='Word')
                        
                        fig.update_traces(textposition='middle center', textfont=dict(size=8))
                        
                        for m, s in enumerate(np.unique(df['Value Cluster']).tolist()):
                            fig.data[m].name = ("Value Cluster "+str(m+1))
                            
                        for s in np.unique(df['Value Cluster']).tolist():
            
                            fig.add_mesh3d(
                                        alphahull= 7,
                                        x=df[df['Value Cluster'] == s]['Dimension 1'],
                                        y=df[df['Value Cluster'] == s]['Dimension 2'],
                                        z=df[df['Value Cluster'] == s]['Dimension 3'],
                                        opacity=0.1,
                                        color=DEFAULT_PLOTLY_COLORS[int(s)]
                                    )
                
            
                fig.layout.template = "plotly_white"
                fig.layout.font = dict(
                            family='Arial',
                        )
    
                fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title_text=('<b>Personality Trait Landscape </b>   |   An overview of attributes associated with ' + str(client_name))
                )
                
                fig.layout.hoverlabel.font = dict(
                            family='Arial',
                            size=8
                            )
                        
                fig.layout.legend=dict(
                            orientation="v",
                            font = dict(size=10),
                            #itemclick="toggleothers",
                            traceorder="normal",
                            xanchor="center",
                            yanchor="bottom",
                            itemsizing="constant"
                        )
            
                primary = "#e6e6e6"
                secondary = "#f0f0f0"
                
                fig.update_xaxes(showgrid=True, gridwidth=0.01, gridcolor=secondary,
                                 showline=True, linewidth=2, linecolor=primary,mirror=True,
                                 zeroline=True, zerolinewidth=2, zerolinecolor=primary)
                fig.update_yaxes(showgrid=True, gridwidth=0.01, gridcolor=secondary,
                                 showline=True, linewidth=2, linecolor=primary,mirror=True,
                                 zeroline=True, zerolinewidth=2, zerolinecolor=primary)
                
                
                # EXPORTING ---------------------------------------------------
                
                name = (title+" - "+str(n_dimensions)+"D - TSNE "+ str(tsne_params)+" - CLUSTER "+ str(cluster_params)+" - STATE "+str(state))
                name = name.replace("}","")
                name = name.replace("{","")
                name = name.replace("'","")
                name = name.replace(","," ")
                name = name.replace(":","")
                
                if as_interactive == True:
                    plotly.offline.plot(fig, filename=(name+".html"), auto_open=False)
                
                if as_picture == True:
                    fig.write_image((name+".png"),
                                    width=900*scaling, 
                                    height=600*scaling,
                                    scale=5
                                    )



# ---


def theme_correlation(df, project_name, cols, n_cols,):

    writer = pd.ExcelWriter(project_name+'_'+str(datetime.today().strftime('%Y%m%d'))+'_Co-Occurence Matrix.xlsx',engine='xlsxwriter')
    workbook=writer.book

    indexes=[df.columns.get_loc(c) for c in cols]

    for i,x in enumerate(indexes):
    
        # THEME-MATRIX CALCULATION
    
        names = df.iloc[:,(indexes[i]+1):(indexes[i]+n_cols[i]+1)].columns.tolist()
    
        theme_matrix = cosine_similarity(df[names].to_numpy().T)
        theme_names = cosine_similarity(df[names].to_numpy().T)
        
        np.fill_diagonal(theme_matrix, 0)
        np.fill_diagonal(theme_names, 1)
    
        # PLOTTING
    
        co_occurence_plot = ff.create_annotated_heatmap(z=theme_matrix,
                                           x=names, 
                                           xgap = 10,
                                           ygap = 10,
                                           y=names,
                                           annotation_text=np.around(theme_names,2),
                                           colorscale = 'greens'
                                           )

        co_occurence_plot.update_layout(height=800,width=2000)
        co_occurence_plot.layout.template = 'plotly_white'
        co_occurence_plot.update_xaxes(showgrid=False, zeroline=False)
        co_occurence_plot.update_yaxes(showgrid=False, zeroline=False)

        # EXCEL WRITING
        
        cooccurence_output = pd.DataFrame(theme_matrix)
        worksheet=workbook.add_worksheet(str(x))
        writer.sheets[str(x)] = worksheet
        cooccurence_output.to_excel(writer,sheet_name=str(x),startrow=0, startcol=0)
            
        plot(co_occurence_plot)
     
    writer.save()


def theme_crosstabs(df, cols, n_cols, project_name, groupings):

    writer = pd.ExcelWriter(project_name+"_"+str(datetime.today().strftime('%Y%m%d'))+'_Theme-CrossTabs.xlsx',engine='xlsxwriter')
    workbook=writer.book

    indexes=[df.columns.get_loc(c) for c in cols]

    for i,x in enumerate(indexes):

        names= df.iloc[:,(indexes[i]+1):(indexes[i]+n_cols[i]+1)].columns.tolist()
        crosstab = (df >> gather('variable', 'value', [names]) >> filter_by(X['value'] != 0))
    
        #pd.crosstab(index=crosstab['Age'], columns=crosstab['variable'])
        
        worksheet=workbook.add_worksheet(cols[i])
        writer.sheets[cols[i]] = worksheet
         
        for w, q in enumerate(groupings):

            # PERCENTAGE CROSSTABS
            percentage_crosstab=pd.crosstab(crosstab[groupings[w]], crosstab["variable"],normalize='index').T
            percentage_crosstab.to_excel(writer,sheet_name=cols[i],startrow=(w*25) , startcol=0)

            # ABSOLUTE CROSSTABS
            absolute_crosstab=pd.crosstab(crosstab[groupings[w]], crosstab["variable"],margins=True).T
            absolute_crosstab.to_excel(writer,sheet_name=cols[i],startrow=(w*25), startcol=(df[groupings[w]].nunique()+7))
    
    writer.save()



def theme_based_wordcloud(df, cols, n_cols, groupings, font_directory, scaling=1,entities = False, phrasing = True, summary = False, width = 1550, split_width = False, space_width = 31.68, height = 1240, exclude_generic_words=True, exclusion_rate=0.02):

        writer = pd.ExcelWriter(project_name+"_"+str(datetime.today().strftime('%Y%m%d'))+'_Word Frequency.xlsx',engine='xlsxwriter')
        workbook=writer.book

        for i,x in enumerate(cols):
            
            # CREATE (SUB-)DIRECTORIES ----------------------------------------
            
            if not os.path.exists(output_directory+"/"+project_name):
                os.mkdir(output_directory+"/"+project_name)
            
            os.chdir(output_directory+"/"+project_name)
            
            if not os.path.exists("./Wordclouds"):
                os.mkdir("./Wordclouds")
                
            os.chdir("./Wordclouds")
            
            if not os.path.exists("./"+str(cols[i])):
                os.mkdir("./"+str(cols[i]))
                
            os.chdir("./"+str(cols[i]))

            # GENERAL WORD FREQUENCIES ----------------------------------------
        
            text = df[cols[i]].tolist()
            text = [string.split(" ") for string in text]
            text = [word for string in text for word in string]

            corpus = text_cleaner(text)

            if phrasing == True:
                corpus = ngram_extract(text,min_freq=4)
        
            wordbag = [word.lower() for word in corpus]
            word_frequencies = dict(Counter(wordbag).most_common(100000000))

            wordcloud = WordCloud(
                    background_color="rgba(255, 255, 255, 0)", 
                    mode="RGBA",
                    mask=None,
                    min_font_size = 15,
                    max_font_size = None,
                    max_words = 100,
                    relative_scaling = 1,
                    #color_func=color_function,
                    font_path=font_directory,
                    width=width*scaling,
                    height=height*scaling)
            
            wordcloud.generate_from_frequencies(frequencies=word_frequencies)
            wordcloud.to_file('Wordcloud - {} - Total.png'.format(x))

            worksheet=workbook.add_worksheet("Total")
            writer.sheets["Total"] = worksheet
            pd.DataFrame.from_dict(word_frequencies,orient='index').to_excel(writer,sheet_name="Total",startrow=0, startcol=0)

            # TOPIC-WISE FREQUENCIES ------------------------------------------

            topic_clusters = df.iloc[:,(indexes[i]+1):(indexes[i]+n_cols[i]+1)].columns.tolist()
            
            for q,topic in enumerate(topic_clusters):

                text = df[(df[topic] == 1) & (df[x].notnull())][cols[i]].tolist()
                text = [string.split(" ") for string in text]
                text = [word for listing in text for word in listing]

                corpus = text_cleaner(text)
                
                if phrasing == True:
                    corpus = ngram_extract(text)

                wordbag = [word.lower() for word in corpus]

                if (exclude_generic_words == True):

                    exclusion_rate=exclusion_rate
                    top_n_words = int(round(len(word_frequencies)*exclusion_rate,0))

                    most_frequent_words={k: word_frequencies[k] for k in list(word_frequencies)[:top_n_words]}
                    most_frequent_words=list(most_frequent_words.keys())

                    wordbag = [word for word in wordbag if word not in most_frequent_words]

                theme_word_frequencies=dict(Counter(wordbag).most_common(100000000))

                if(split_width == True):
                    width=int((900.16-((len(unique_values)-1)*space_width))/len(unique_values))

                wordcloud = WordCloud(
                        background_color="rgba(255, 255, 255, 0)", 
                        mode="RGBA",
                        mask=None,
                        min_font_size = 15,
                        max_font_size = None,
                        max_words = 100,
                        relative_scaling = 0.9,
                        #color_func=byw_grey,
                        font_path=font_directory,
                        width=width*scaling,
                        height=height*scaling)

                wordcloud.generate_from_frequencies(frequencies=theme_word_frequencies)
                wordcloud.to_file("Wordcloud - {} - Theme '{}'.png".format(x,topic))

                worksheet=workbook.add_worksheet(topic)
                writer.sheets[topic] = worksheet
                pd.DataFrame.from_dict(theme_word_frequencies,orient='index').to_excel(writer,sheet_name=topic,startrow=0, startcol=0)

        writer.save()

            # # TEXT SUMMARIZATION --------------------------------------------------
            
            # if summary == True:
                
            #     from gensim.summarization.summarizer import summarize
            #     from gensim.summarization import keywords
                
            #     print(keywords(corpus).split('\n'))
            #     print(summarize(corpus, ratio=0.05))
                
            
            # ## ENTITIES -----------------------------------------------------------
            
            # if entities == True:

            #     doc = nlp(corpus)
                
            #     for label in set([w.label_ for w in doc.ents]): 
            #         entities = [e.string.lower().strip() for e in doc.ents if label==e.label_] 
            #         entities = list(set(entities)) 
            #         print(label,entities)
        
            #     ## MENTIONS FOR SPECIFIC WORDS ---------------------------------------------
                
            #         for entity in entities:
            #             print(i,x, label, entity)
            #             print(pos_words(doc, [entity], "ADJ"))



def group_based_wordcloud(df, cols, n_cols, groupings, font_directory, scaling=1, phrasing=False):

    df["Total"]=1
    groupings.append("Total")
    df = df.fillna(0)

    writer = pd.ExcelWriter(project_name+"_"+str(datetime.today().strftime('%Y%m%d'))+'_Word Frequency per Group.xlsx',engine='xlsxwriter')
    workbook=writer.book
    
    for i,x in enumerate(cols):
    
        # CREATE (SUB-)DIRECTORIES ----------------------------------------
        
        if not os.path.exists(output_directory+"/"+project_name):
            os.mkdir(output_directory+"/"+project_name)
        
        os.chdir(output_directory+"/"+project_name)
        
        if not os.path.exists("./Wordclouds"):
            os.mkdir("./Wordclouds")
            
        os.chdir("./Wordclouds")
        
        if not os.path.exists("./"+str(cols[i])):
            os.mkdir("./"+str(cols[i]))
            
        os.chdir("./"+str(cols[i]))

        # GROUP-SPECIFIC WORDCLOUDING ---
    
        for var in groupings:
            
            unique_values = [value for value in df[var].unique().tolist() if value not in [np.NaN,'-1']]
        
            for item in unique_values:

                text = df[(df[var] == item) & (df[x].notnull())][cols[i]].tolist()
                text = [string.split(" ") for string in text]
                text = [word for listing in text for word in listing]

                corpus = text_cleaner(text)

                if phrasing == True:
                    corpus = ngram_extract(text)

                wordbag = [word.lower() for word in corpus]

                group_word_frequencies=dict(Counter(wordbag).most_common(100000000))
                wc_width=int((900.16-((len(unique_values)-1)*31.68))/len(unique_values))
                
                wordcloud = WordCloud(
                        background_color="rgba(255, 255, 255, 0)", 
                        mode="RGBA",
                        mask=None,
                        #min_font_size = 30,
                        max_font_size = None,
                        max_words = 10000,
                        relative_scaling = 1,
                        #color_func=color_function,
                        font_path=font_directory,
                        width=wc_width*scaling,
                        height=int(240.96*scaling))
              
                wordcloud.generate_from_frequencies(frequencies=group_word_frequencies)
                wordcloud.to_file("Wordcloud - {} - {}: '{}'.png".format(x,var,str(item)).replace("/"," "))

                worksheet=workbook.add_worksheet((str(var)+" - "+str(item)))
                writer.sheets[(str(var)+" - "+str(item))] = worksheet
                pd.DataFrame.from_dict(group_word_frequencies,orient='index').to_excel(writer,sheet_name=(str(var)+" - "+str(item)),startrow=0, startcol=0)
                
                print(item, var, group_word_frequencies)
            
        writer.save()



def translating(key_directory, df, target_language = 'en'):
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_directory
    
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
                        translation = translate_client.translate(to_trans[i], target_language=target_language)
                        df[column][i] = translation['translatedText']
                        print("T",to_trans[i])
                        translates.append(to_trans[i])
                        
                    except:
                        df[column][i]=to_trans[i]
                        print("NT",to_trans[i])    
                        
    return df, translates

