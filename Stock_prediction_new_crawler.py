#import library
from selenium import webdriver
import pandas as pd
import datetime
from newspaper import Article
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import gensim
import numpy as np
import CNN_Stock_Prediction_train
import random
import matplotlib.pyplot as plt

# 1. google finance page url with 'GOOGL' searching word to gather the news url
def get_url_from_google_finance(url_searching_page,searching_page):
    #downloading news
    driver = webdriver.Chrome('C:/Users/user/chromedriver')
    driver.implicitly_wait(3)
    #get the url for AGOOGL in searching page
    driver.get(url_searching_page)
    links_list=[]
    for j in range(searching_page):
        for i in range(1, 11, 1):
            news_box = driver.find_element_by_xpath('//*[@id="news-main"]/div[%s]/span'%i)
            news_link = news_box.find_element_by_tag_name('a')
            news_href = news_link.get_attribute('href')
            links_list.append(news_href)
        page_box = driver.find_elements_by_class_name('nav_b')
        page_link = page_box[len(page_box)-1].find_element_by_tag_name('a')
        page_link.click()
    return links_list

# scrap the contents of news
# input : url_list
# output : dataframe news text.
def get_news_content(links_list):
    news_df = pd.DataFrame()
    for url in links_list:
        try:
            article = Article(url,language='en')
            article.download()
            article.parse()

            news_title = article.title
            news_datetime = article.publish_date
            news_date = datetime.datetime.strptime(news_datetime.strftime('%Y-%m-%d %H:%M:%S')[:10],'%Y-%m-%d')
            news_text = article.text
            news_df = news_df.append(pd.DataFrame(index=[news_date], data={'title': news_title, 'text': news_text}))
        except:
            continue
    news_df = news_df.dropna(how='any')
    news_df = news_df.sort_index()
    remove_list=[]
    for i,text in enumerate(news_df['text']):
        if text=='':
            remove_list.append(i)
    news_df.drop(news_df.index[remove_list],inplace=True)
    return news_df

# For LDA make the text with tokenized, normalized and stemmed.
def preprocessing_lda(documnets):
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = get_stop_words('en')
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    doc_set = []
    for doc in documnets:
        doc_set.append(doc)

    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts

if __name__ == "__main__":
    # first, make the input data - training and testing
    # for making LDA training data set, scrap the google finance news about "GOOGL" company.
    urls=get_url_from_google_finance('https://www.google.com/finance/company_news?q=NASDAQ%3AGOOGL&ei=NcKIWbHgLouD0gSDubbADA',searching_page=18)
    news_df=get_news_content(urls)
    
    # making the stemmed text.
    texts=preprocessing_lda(news_df['text'])
    # for LDA making dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # I think the news can have the 3 minimum group.(good new for stock, bad news for stock, meaningless news for stock)
    # if the news are too many, we can get the more than 3 group.
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, update_every=3,
                                               chunksize=10000, passes=100)
    # show the topic for each group.
    topics = ldamodel.print_topics(3, num_words=20)
    topics_matrix = ldamodel.show_topics(formatted=False, num_words=20)
    for i in range(3):
        topics_array = np.array(topics_matrix[i][1])
        print([str(word[0]) for word in topics_array])
        print()
    
    # each news should be classified for their each group.
    news_df['lda_category'] = -1
    news_df['lda_pro'] = -1.0
    for i,one_doc in enumerate(texts):
        bow = dictionary.doc2bow(one_doc)
        t = ldamodel.get_document_topics(bow)
        category=0
        max_pro=0.0
        for j in t:
            if max_pro<j[1]:
                category = j[0]
                max_pro = j[1]

        news_df.set_value(news_df.index[i],'lda_category',category)
        news_df.set_value(news_df.index[i], 'lda_pro', max_pro)
    
    # the LDA group is just trend. In my thought, the stock price can be determined by many things.
    # we should mapping days and the stock price of tommorrow
    googl_stock = pd.read_csv('./data/daily-stock/googl_.csv',index_col=0, encoding='utf-8')
    googl_stock = googl_stock.sort_index(ascending=True)
    googl_stock['pct_change'] = googl_stock.pct_change().shift(-1).dropna(how='any')
    news_df['pct_chaneg']=0.0
    for i,news_date in enumerate(news_df.index):
        try:
            news_df.set_value(news_df.index[i], 'pct_chaneg', googl_stock.loc[news_date._date_repr]['pct_change'])
        except:
            try:
                news_df.set_value(news_df.index[i], 'pct_chaneg', googl_stock.loc[(news_date+datetime.timedelta(days=-1))._date_repr]['pct_change'])
            except:
                try:
                    news_df.set_value(news_df.index[i], 'pct_chaneg', googl_stock.loc[(news_date + datetime.timedelta(days=-2))._date_repr]['pct_change'])
                except:
                    continue
    news_df = news_df.dropna()
    # now we can see the which group has many good news for stock price and which has bed news.
    avg_list=[]
    for i in range(5):
        avg_list.append(news_df[news_df['lda_category'] == i]['pct_chaneg'].mean())
    argmin = avg_list.index(min(avg_list))
    argmax = avg_list.index(max(avg_list))
    
    # i save the raw text data and stemmed text data for CNN input.
    cleaned_text_neg = preprocessing_lda(news_df[news_df['lda_category'] == argmin]['text'])
    with open("./data/rt-polarity/rt-polarity-googl-stem.neg", "a",encoding='utf-8') as output:
        for text_neg in cleaned_text_neg:
            for word in text_neg:
                output.write(str(word+" "))

    cleaned_text_pos = preprocessing_lda(news_df[news_df['lda_category'] == argmax]['text'])
    with open("./data/rt-polarity/rt-polarity-googl-stem.pos", "a",encoding='utf-8') as output:
        for text_pos in cleaned_text_pos:
            for word in text_pos:
                output.write(str(word + " "))

    text_neg = news_df[news_df['lda_category'] == argmin]['text']
    with open("./data/rt-polarity/rt-polarity-googl.neg", "a", encoding='utf-8') as output:
        for word in text_neg:
            output.write(word)

    text_pos = news_df[news_df['lda_category'] == argmax]['text']
    with open("./data/rt-polarity/rt-polarity-googl.pos", "a", encoding='utf-8') as output:
        for word in text_pos:
            output.write(word)
    
    #stemmed text data result is not good.
    #random searching for main hyperparameter
    # random_best_score = 0
    # random_scores = []
    #
    # embedding_dim_list = []
    # filter_sizes_lists = []
    # num_filters_list = []
    # for i in range(100):
    #     embedding_dim = random.randint(128,1024)
    #     filter_sizes = random.randint(3,10)
    #     filter_sizes_list = [filter_sizes,filter_sizes+1,filter_sizes+2]
    #     num_filters = random.randint(128,1024)
    #     score = CNN_Stock_Prediction_train.train("rt-polarity-googl-stem.pos", "rt-polarity-googl-stem.neg", embedding_dim, filter_sizes_list, num_filters)
    #
    #     embedding_dim_list.append(embedding_dim)
    #     filter_sizes_lists.append(filter_sizes_list)
    #     num_filters_list.append(num_filters)
    #     random_scores.append(score)
    #
    #     if score > random_best_score:
    #         random_best_score = score
    #         best_embedding_dim = embedding_dim
    #         best_filter_sizes_list = filter_sizes_list
    #         best_num_filters = num_filters
    # print("best hyperparameter: score[%s], embedding_dim[%s],filter_size[%s],num_filter[%s]"%(random_best_score,best_embedding_dim,best_filter_sizes_list,best_num_filters))
    # embedding_result = pd.DataFrame(index=embedding_dim_list, data=random_scores)
    # embedding_result = embedding_result.sort_index()
    #
    # plt.title('Random_Search')
    # plt.plot(embedding_result, 'p--', markersize=8, label=u'Random_Search', color='c')
    # plt.xlabel('embedding', fontdict={'size': 20})
    # plt.ylabel('score', fontdict={'size': 20})

    random_best_score = 0
    random_scores = []

    embedding_dim_list = []
    filter_sizes_lists = []
    num_filters_list = []
    for i in range(100):
        embedding_dim = random.randint(128, 256)
        filter_sizes = random.randint(3, 10)
        filter_sizes_list = [filter_sizes, filter_sizes + 1, filter_sizes + 2]
        num_filters = random.randint(128, 256)
        print("embedding_dim : %s, filter_sizes_list : %s,num_filters : %s"%(embedding_dim,filter_sizes_list,num_filters))
        score = CNN_Stock_Prediction_train.train("./data/rt-polarity/rt-polarity-googl.pos", "./data/rt-polarity/rt-polarity-googl.neg",
                                                 embedding_dim, filter_sizes_list, num_filters)

        embedding_dim_list.append(embedding_dim)
        filter_sizes_lists.append(filter_sizes_list)
        num_filters_list.append(num_filters)
        random_scores.append(score)

        if score > random_best_score:
            random_best_score = score
            best_embedding_dim = embedding_dim
            best_filter_sizes_list = filter_sizes_list
            best_num_filters = num_filters
    print("best hyperparameter: score[%s], embedding_dim[%s],filter_size[%s],num_filter[%s]" % (random_best_score, best_embedding_dim, best_filter_sizes_list, best_num_filters))
    embedding_result = pd.DataFrame(index=embedding_dim_list, data=random_scores)
    embedding_result = embedding_result.sort_index()

    plt.title('Random_Search')
    plt.plot(embedding_result, 'p--', markersize=8, label=u'Random_Search', color='c')
    plt.xlabel('embedding', fontdict={'size': 20})
    plt.ylabel('score', fontdict={'size': 20})
    
    # conclusion.
    # the result of this model is just good but not reached my expectation.
    # the google finance news has just little news.
    # i scrap the naver news for sk Hynics 
    # there are more than 3000 news from 2012 years to now.
    # using korean stop words i build the same model.
    # at that moment, i change the code little. i seperate training data and test data totally.
    # In above model, train data and test data is mixed like bootstrap. this can cause that the test accuracy is so good. because the ML can remember the training data and test data.
    # for testing the prediction, not memory, i seperate training data and test data. 
    # the traning data would be used for traning only and test data too.
    # But result is still bed. Accually really reallly bed.(the mean of performance(accuracy) is just about 25%)
    # when i read the CNN text classifier, the accuracy can be about 0.8.
    # i cannot find out what is really worng.
    # please help me siraj, give me some advices.
    # i really really really want to make the prediction for stock price and ML for stock.