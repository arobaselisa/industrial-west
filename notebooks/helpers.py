import unidecode as unidecode
import pandas as pd
import tomotopy as tp
import time
import spacy

def load_preprocess_df(data_path, start_date, end_date, word_list, journal):
    if "de" in data_path:
        data = pd.read_csv(data_path, index_col="id")
    else:
        data = pd.read_csv(data_path, index_col="Unnamed: 0") 
    if "it" in data_path:
        data["publication_date"] = pd.to_datetime(data["publication_date"], format='%Y')
    else:
        data["publication_date"] = pd.to_datetime(data["publication_date"])
    data = data[(data["publication_date"]>= start_date) & (data["publication_date"]<end_date)]
    data = data.drop_duplicates(subset=["content"], keep = "first")
    data = data.dropna()
    data = data[data['content'].map(len) < 500000] # remove too long articles
    data["content"] = data.apply(lambda row: extract_word_article(row.content, word_list, journal), axis=1)
    return data


def extract_word_article(whole_text, word_list, journal):
    if journal == "Le Figaro" or journal == "Imparcial":
        keyword_text = ""
        for part in whole_text.split('\n'):
            if all(text in unidecode.unidecode(str.lower(part)) for text in word_list):
                keyword_text += " " + part
        return keyword_text
    elif journal == "Herald":
        lines = whole_text.split('\n')
        titles_ix, word_ix = get_word_title_idx(lines, word_list[0])
        boundaries = []
        for c_ix in word_ix:
            greater_titles = [idx for idx in titles_ix if idx > c_ix]
            end_title = titles_ix.index(greater_titles[0]) if len(greater_titles) else -1
            start_title = titles_ix.index(greater_titles[0])-1 if len(greater_titles) else len(titles_ix) - 1
            if titles_ix[start_title] == titles_ix[end_title]:
                return whole_text
            boundaries += [(titles_ix[start_title], titles_ix[end_title])]
        text = " ".join([" ".join(lines[boundaries[i][0]: boundaries[i][1]]) for i in range(len(boundaries))])
        return text[:int(len(text)/text.count(text[:15]))]
    else:
        return whole_text

def get_word_title_idx(lines, word):
    title_idx = [0]
    word_idx = []
    for i, l in enumerate(lines):
        if word in l.lower():
            word_idx += [i]
        if len(l) > 4:
            if len([c for c in list(l) if c.isalpha()])<1/3*len(l):
                title_idx += [i]
            elif len([c for c in list(l) if c.isupper()])>2/3*len(l):
                title_idx += [i]
    return title_idx, word_idx


def create_corpus(data, language):
    raw_articles = data["content"].to_list()
    sp = spacy.load(language, disable=["ner",  "entity_linker",   "parser", 
                                           "textcat", "textcat_multilabel",  "senter",  "sentencizer",  "transformer"
                                          ]) ## you can load any language now and it will automatically decide on the stop words
    sp.max_length = 2327128
    start= time.time()
    raw_docs = []
    num_articles = len(raw_articles)
    for i, doc in enumerate(raw_articles):
        if i%500 == 1:
            print("Runtime: %.2f seconds" %(time.time() - start), "|| Completed: %s of %s" %(i, num_articles))
        raw_docs.append(preprocess_text(sp, doc))

    corpus = tp.utils.Corpus()
    for doc in raw_docs:
        if doc:
            corpus.add_doc(doc)
            
    return raw_docs, corpus

def preprocess_text(sp, text: str, user_data = None): ### takes text as a string (not list) and return list
    text = text.lower()
    text = [word.lemma_ for word in sp(text) if word.is_alpha and (not word.is_stop) and len(word)>3]  
    ### above: removes punctuation, digits, stop words, lemmatizes words
    return text

   