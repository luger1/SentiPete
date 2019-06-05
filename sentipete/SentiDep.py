import spacy
import re
import os
import pickle
import numpy as np
import pandas as pd
from germalemma import GermaLemma
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm


class SentiDep:

    def __init__(self, **kwargs):
        """
            Sentiment-Analyzer for german texts.
            Get the polarity values of words depending on
            polarity values of associated descriptive words
            e.g. 'das schöne Wetter' -> polarity of 'Wetter' == polarity of 'schöne'

            Purpose: find out in which sentiment context your keywords appear in a text.
            Note: Works with spacy, nltk and germalemma
        """
        sentiws_path = kwargs.get('sentiws_file', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                               "data/sentiws.pickle"))
        polarity_mod_path = kwargs.get('polarity_modifiers_file',
                                       os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "data/polarity_modifiers.pickle"))
        negations_path = kwargs.get('negations_file',
                                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "data/negationen_lexicon.pickle"))
        stts_path = kwargs.get('stts_file',
                               os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data/stts.pickle"))
        self.sentiws = pickle.load(open(sentiws_path, 'rb'))
        self.polarity_modifications = pickle.load(open(polarity_mod_path, 'rb'))
        self.negations = pickle.load(open(negations_path, 'rb'))
        self.nlp = spacy.load("de_core_news_md")
        self.germalemmatizer = GermaLemma()
        self.stts = pickle.load(open(stts_path, 'rb'))
        self.german_stops = stopwords.words('german')

    def tokenize(self, text):
        """
        Tokenize a string using spacy's tokenizer.
        Input: text/string
        Output: spacy_doc
        """
        return self.nlp(text)

    def sentiws_spacy_tag_mapper(self, pos_tag, **kwargs):
        """
        Function for mapping SentiWS POS-tags to spacy POS-tags and reverse.
        Input: pos_tag, optional: direction
               -> values: 1 (sentiws to spacy), -1 (spacy to sentiws)
               -> default: 1
        Output: python str
        """
        direction = kwargs.get('direction', 1)
        senti_map = {"ADJX": "ADJ", "ADV": "ADV", "NN": "NOUN", "VVINF": "VERB"}
        if direction > 0:
            return senti_map[pos_tag]
        elif direction < 0:
            return {value: key for key, value in senti_map.items()}[pos_tag]

    def get_polarity(self, word, pos_tag):
        """
        Getter Function for retaining the polarity value by SentiWS for a certain word with POS-tag.
        Input: word, pos_tag
        Output: tuple(word, polarity-value, pos_tag)
        """
        senti_words = list(filter(
            lambda x: x[0] == word and self.sentiws_spacy_tag_mapper(x[2]) == pos_tag, self.sentiws)
        )
        if senti_words:
            senti_words = sorted(senti_words, key=lambda y: y[1] ** 2, reverse=True)[0]
            return senti_words

    def modify_polarity(self, child, polarity):
        """
        Function to consider polarity enhancer and reducer.
        Input: token.text, token.child.text, token.pos_ (of word)
        Output: tuple(word, polarity-value, pos_tag)
        """
        senti_word = polarity
        if senti_word:
            if child in self.polarity_modifications["polarity_enhancer"]:
                return (senti_word[0], senti_word[1] * 1.5, senti_word[2])
            elif child in self.polarity_modifications["polarity_reducer"]:
                return (senti_word[0], senti_word[1] * 0.5, senti_word[2])

    def easy_switch(self, word):
        """
        Function for finding depending negations without dealing with complex issues.
        Input: token/word
        Output: True/False
        """
        neg_search = [re.search(r'%s' % (n), word) for n in self.negations["negation_regex"]]
        neg_search = list(filter(lambda z: z != None, neg_search))
        return bool(neg_search)

    def add_polarities(self, list_of_polarity_tuples):
        """
        Summing up a list of polarity-tuples
        :param list_of_polarity_tuples:
        :return: polarity value -> float
        """
        all_pols = [lpt[1] for lpt in list_of_polarity_tuples]
        return sum(all_pols)

    def calc_parent_polarity(self, spacy_token, token_polarity, children_polarities):
        """
        Calculating the parent polarity value depending on the children polarities
        :param spacy_token:
        :param token_polarity:
        :param children_polarities:
        :return: parent_polarity -> tuple(word, polarity, POS-tag)
        """
        if token_polarity and children_polarities:
            added_children_polarities = self.add_polarities(children_polarities)
            if added_children_polarities > 0:
                token_polarity = (spacy_token.text,
                                  token_polarity[1] + added_children_polarities,
                                  spacy_token.pos_)
            elif added_children_polarities < 0:
                token_polarity = (spacy_token.text,
                                  (token_polarity[1] + (-1 * added_children_polarities)) * (-1),
                                  spacy_token.pos_)
        elif not token_polarity and children_polarities:
            token_polarity = (spacy_token.text,
                              self.add_polarities(children_polarities),
                              spacy_token.pos_)
        return token_polarity

    def switch_polarity(self, polarity, spacy_doc_sent):
        """
        Switching polarity value depending on negation context of whole sentence.
        Classic negation (kein, nicht, ...) are recognized as well as
        negation stops (aber, obwohl, ...)
        :param polarity:
        :param spacy_doc_sent:
        :return: tuple(word, polarity, POS-tag, negation: boolean)
        """
        negation_trigger = False
        for i, token in enumerate(spacy_doc_sent):
            for negex in self.negations['negation_regex']:
                regex = r'%s' % (negex)
                negation_search = re.search(regex, token.text, re.I)
                if negation_search:
                    negation_trigger = not negation_trigger
            if token.lower_ in self.negations['polarity_switches']:
                if token.text == '.':
                    if token.pos_ == 'PUNCT':
                        negation_trigger = not negation_trigger
                    else:
                        continue
                else:
                    negation_trigger = not negation_trigger
            if token.text == polarity[0]:
                if negation_trigger:
                    negated_polarity = (polarity[0], -polarity[1],
                                        polarity[2], "negation: " + str(negation_trigger))
                else:
                    negated_polarity = (polarity[0], polarity[1],
                                        polarity[2], "negation: " + str(negation_trigger))
                return negated_polarity

    def get_depending_polarities(self, text, keywords):
        """
        Get keyword associated polarity values of german texts.
        Polarity analysis including polarity reducer/enhancer and negations
        :param text:
        :param keywords:
        :return: Context-polarity value of keywords -> list of tuples
        """
        spacy_doc = self.nlp(text, disable=['ner', 'textcat'])
        parent_polarities = []
        keywords = [k.lower() for k in keywords]
        for sent in spacy_doc.sents:
            for i, token in enumerate(sent):
                token_polarity = self.get_polarity(token.text, token.pos_)
                children_polarities = []
                if token.lower_ in keywords:
                    children = token.children
                    if children:
                        for child in children:
                            child_polarity = self.get_polarity(child.text, child.pos_)
                            if child_polarity:
                                children_polarities.append(child_polarity)
                    parent_polarity = self.calc_parent_polarity(token, token_polarity, children_polarities)
                    if parent_polarity:
                        modified_parent_polarities = []
                        for child in children:
                            modified_parent_polarities.append(self.modify_polarity(child, parent_polarity))
                        added_modified_parent_polarity = None
                        if modified_parent_polarities:
                            added_modified_parent_polarity = self.add_polarities(modified_parent_polarities)
                        if added_modified_parent_polarity:
                            added_modified_parent_polarity = (token.text,
                                                              added_modified_parent_polarity,
                                                              token.pos_ + "_modified")
                            parent_polarities.append(self.switch_polarity(
                                added_modified_parent_polarity, sent))
                        else:
                            parent_polarities.append(self.switch_polarity(parent_polarity, sent))
        parent_polarities = [(term.lower(), t_pol, t_pos, neg) for term, t_pol, t_pos, neg in parent_polarities]
        return parent_polarities

    def lemmatize(self, spacy_token):
        """
        Lemmatizer using stts-tagset, spacy-token and GermaLemma.
        Input: spacy token -> german model
        Output: python str
        """
        tag = spacy_token.tag_
        if tag.startswith(('N', 'V', 'ADJ', 'ADV')) and tag in self.stts:
            return self.germalemmatizer.find_lemma(spacy_token.text, tag)
        else:
            return spacy_token.text

    def generate_topics(self, texts, num_topics=10):
        """
        Generate a list with 30 most frequent nouns in a text.
        Input: text -> len(text) <= 50000
        Output: nltk.FreqDist-object
        """
        tokens = [[token for token in self.tokenize(text)] for text in texts]
        tokens = [[self.lemmatize(t) for t in token if t.pos_ == 'NOUN'\
                  and t.lower_ not in self.german_stops] for token in tokens]
        docs = [" ".join(t) for t in tokens]
        cv = CountVectorizer(max_df=0.85, max_features=10000)
        word_count_vector = cv.fit_transform(docs)
        tf = TfidfTransformer(smooth_idf=True, use_idf=True)
        tf.fit(word_count_vector)
        feature_names = cv.get_feature_names()
        tf_idf_scores = []
        for doc in docs:
            cv_vector = cv.transform([doc])
            tf_idf_vector = tf.transform(cv_vector)
            sorted_items = self.sort_coo(tf_idf_vector.tocoo())
            keywords, scores = self.extract_topn_from_vector(feature_names, sorted_items, 10)
            tf_idf_scores += list(zip(keywords, scores))

        tfidf_topics = sorted(tf_idf_scores, key=lambda x: x[1], reverse=False)
        return dict(tfidf_topics[:num_topics])

    def sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def extract_topn_from_vector(self, feature_names, sorted_items, topn=10):
        sorted_items = sorted_items[:topn]
        score_vals = []
        feature_vals = []

        for idx, score in sorted_items:
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]
        return results, score_vals

    def create_clinic_polarity_dict(self, key_list, topics):
        """
        Compute polarity scores document-wise
        :param key_list: list of polarity-scores and document-key
                         -> form: [[polarity-scores_1, document-key_1] ...]
                         -> hint: simple pandas dump with
                            df[[polarity-values, document]].values.tolist()
        :param topics: list of keywords associated with a certain topic
        :return: polarities_dict in form:
                 {document_key_1: polarities_1, ...}
        """
        polarities = {}
        clinic_counter = {}
        for rl in tqdm(key_list):
            if not rl[1] in clinic_counter.keys():
                clinic_counter[rl[1]] = 1
            key = f'{rl[1]}_{clinic_counter[rl[1]]}'
            polarities[key] = self.get_depending_polarities(rl[0], topics)
            clinic_counter[rl[1]] += 1
        return polarities

    def create_polarity_df(self, polarities, topics):
        """
        Transforms polarity-scores from 'create_clinic_polarity_dict' output
        to a formatted pandas dataframe
        :param polarities: polarities-dict (output from 'create_clinic_polarity_dict')
        :param topics: list of keywords associated with a certain topic
        :return: polarity_df (formatted pandas dataframe) of form:
                 columns: keywords/topics
                 rows: document-keys
                 values: float(polarity-scores) or np.nan
        """
        filtered_polarities = [(clinic, polarity) for clinic, polarity in polarities.items() if polarity]
        columns = {t: [] for t in topics}
        ids = {"Klinik": []}
        for clinic, polarity in tqdm(filtered_polarities):
            ids["Klinik"].append(clinic)
            row = {t: [] for t in topics}
            for pol in polarity:
                row[pol[0].lower()] = pol[1]
            for word, p in row.items():
                if not p:
                    columns[word].append(np.nan)
                else:
                    columns[word].append(p)
        for key, value in columns.items():
            if len(value) < len(ids["Klinik"]) or len(value) > len(ids["Klinik"]):
                raise ValueError("Values in dict must have same length!")

        polarity_df = pd.DataFrame(data=columns, index=ids["Klinik"])
        return polarity_df

    '''
        def prepare_for_polarityplot(self, list2d_of_polarities):
            """
            Prepares a list of document-polarities for polarity-poltting
            :param list2d_of_polarities: list of outputs from get_depending_polarities-function
            :return: - sentiment_df -> essential for polarity-plot
                                    -> contains mean-polarity-values for each keyword
                                    -> col: keywords
                                    -> row: mean-polarity-value of all terms in docs
                     - len_dict(=nobs-dict) -> optional for polarity-plot
                     showing number of observations (nobs)
            """
            topics_dict = {}
            for pol in list2d_of_polarities:
                for p in pol:
                    if p:
                        if not p[0] in topics_dict.keys():
                            topics_dict[p[0]] = []
                        topics_dict[p[0]].append((p[1], p[2], p[3]))
            norm_dict = {}
            ld = {}
            max_len = self.max_sentiment(topics_dict)
            for term, value in topics_dict.items():
                norm_dict[term] = [v[0] for v in value] + [np.nan] * (max_len - len(value))
                ld[term] = len(value)
            mean_dict = norm_dict
            len_dict = ld
            sentiment_df = pd.DataFrame(data=mean_dict)
            return sentiment_df, len_dict

        def max_sentiment(self, topics_dict):
            return len(max(topics_dict.items(), key=lambda x: len(x[1]))[1])

        '''


if __name__ == '__main__':
    from .SentiPlotting import SentiPlotting
    from tqdm import tqdm

    rka = ["RHÖN-KLINIKUM_Campus_Bad_Neustadt", "Zentralklinik_Bad_Berka_GmbH",
           "Universitätsklinikum_Gießen_(Justus-Liebig-Universität)",
           "Universitätsklinikum_Marburg_(Philipps-Universität)",
           "Klinikum_Frankfurt_Oder_Markendorf"]
    single_ratings = pd.read_excel("../Klinikbewertungen/Klinikbewertung_Einzelbewertungen.xlsx")
    single_reports = single_ratings[single_ratings["Klinik"].isin(rka)]
    sd = SentiDep(sentiws_file="../sentiws.pickle", polarity_modifiers_file="../polarity_modifiers.pickle",
                  negations_file="../negationen_lexicon.pickle", stts_file="../stts.pickle")
    sp = SentiPlotting()
    topic_corpus = single_reports["Erfahrungsbericht"].values.tolist()
    topics = sd.generate_topics(topic_corpus, 10)
    print("Topics:")
    [print(t) for t in topics.items()]
    reports_list = single_reports[["Erfahrungsbericht", "Klinik"]].values.tolist()
    polarities = sd.create_clinic_polarity_dict(reports_list, topics)
    polarity_df = sd.create_polarity_df(polarities, topics)
    intervals_df = sp.prepare_for_polarityplot(polarity_df)
    sp.polarityplot_lickert(intervals_df,
                            title="Themen-abhängige Sentiment-Analyse für die RKA-Kliniken",
                            figure_save_path="../RKA_Polaritäten_Lickertplot.png")
