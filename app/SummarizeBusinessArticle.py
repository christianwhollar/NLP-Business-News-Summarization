from bs4 import BeautifulSoup
import re
import requests
import streamlit as st
from summarizer import GenSummarizer, ExtSummarizer

# Ref Article : https://www.huffpost.com/entry/fedex-startup-conference-west-memphis-three_n_2648977

st.set_page_config(layout='centered')

if 'clicks' not in st.session_state:
    st.session_state['clicks'] = {}
    st.session_state.clicks['download_button'] = False
    st.session_state.clicks['summarize_button'] = False

def click(key):
    st.session_state.clicks[key] = True

def unclick(key):
    st.session_state.clicks[key] = False

st.title('Business Article Summarizer')

text = ''

url_mode = st.select_slider(label = '', options = ['Custom URL', 'Trial URLs'])

col1, col2 = st.columns([4, 1])

with col1:
    if url_mode == 'Custom URL':
        url = st.text_input('Type URL:')

    if url_mode == 'Trial URLs':
        url = st.selectbox(
            'Select a URL:',
            (
                "https://www.huffingtonpost.com/entry/american-apparel-shoplifting_us_5bb2b62ee4b0480ca65a2b07",
                "https://www.huffingtonpost.com/entry/leadership-eavesdropping_us_5bb2b62de4b0480ca65a2a90",
                "http://dealbook.nytimes.com/2012/02/18/bonds-backed-by-mortgages-regain-allure/"
            )
        )

with col2:
    st.markdown('')
    download_button = st.button('Download Article', on_click = click, args = ['download_button'])

if st.session_state.clicks['download_button']:
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Extract body text from article
    title = soup.find('title')
    title = [i.text for i in title]
    title = ' '.join(title)
    bodytext = soup.find_all('p')
    bodytext = [i.text for i in bodytext]
    text = ' '.join(bodytext)

    st.subheader(title)
    st.write(text)

    model = st.radio('Model Selection', options = ['Bart', 'T5', 'TextRank'])
    summarize_button = st.button('Summarize', on_click = click, args = ['summarize_button'])

if st.session_state.clicks['summarize_button']:
    
    if model == 'Bart':
        st.subheader('Bart Summary')

        gsm = GenSummarizer()
        
        with st.spinner('Generating summary...'):
            summary = gsm.generate_summary('bart_model', text)
            
        pattern = r'(?<=[.?!])(?!\s*")'
        summary = re.sub(pattern, ' ', summary)

        sentences = gsm.split_sentences(summary)
        for sentence in sentences:
            if any(char.isalpha() for char in sentence):
                st.markdown(f'-{sentence}')

    if model == 'T5':
        st.subheader('T5 Summary')

        gsm = GenSummarizer()
        
        with st.spinner('Generating summary...'):
            summary = gsm.generate_summary('T5_model', text)

        pattern = r'(?<=[.?!])(?!\s*")'
        summary = re.sub(pattern, ' ', summary)
        
        sentences = gsm.split_sentences(summary)
        for sentence in sentences:
            if any(char.isalpha() for char in sentence):
                st.markdown(f'-{sentence}')

    if model == 'TextRank':
        st.subheader('TextRank Summary')
        esm = ExtSummarizer()

        ext_sents = esm.get_sentences(text)
        processed_article = esm.preprocess(ext_sents)  
        fvs = esm.vectorize(processed_sents = processed_article)
        adj_mat = esm.generate_adjacency_matrix(feature_vecs = fvs)
        summary = esm.summarize(ext_sents, adj_mat, top_n = 3)

        pattern = r'(?<=[.?!])(?!\s*")'
        summary = re.sub(pattern, ' ', summary)
        
        sentences = esm.get_sentences(summary)
        for sentence in sentences:
            if any(char.isalpha() for char in sentence):
                st.markdown(f'-{sentence}')

