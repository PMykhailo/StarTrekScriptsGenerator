import streamlit as st
import pandas as pd
from os.path import dirname
from transformers import pipeline
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer


@st.cache_resource()
def load_generator(path, _tkn):
    model = GPT2LMHeadModel.from_pretrained(path)
    return pipeline('text-generation', model=model, tokenizer=_tkn, device=model.device)

@st.cache_resource()
def load_tokenizer():
    return GPT2Tokenizer.from_pretrained('gpt2')

@st.cache_data()
def load_all_lines(path):
    return pd.read_csv(path)



st.title('This is GPT-2 model fine-tuned with Star Trek scipts')
st.markdown('''Write a line for  model to begin  with and it will try it's best to continue it as Star Trek script. Please write caracter name in CAPS.''')

tokenizer = load_tokenizer()
generator = load_generator(f'''{dirname(__file__)}/models/all_scripts''', tokenizer)
all_lines = load_all_lines(f'{dirname(__file__)}/STscripts/all_lines')

if 'input_val' not in st.session_state:
    st.session_state['input_val'] = '''PICARD: You will agree, Data, that Starfleet's orders are difficult?'''

def random_val():
    st.session_state['input_val'] = all_lines.sample(n=1)['0'].item()

random_line = st.button('Start with random line from scripts database', on_click=random_val)



with st.form(key = 'parametres'):
    input = st.text_input('Enter your own starting line here',value=st.session_state['input_val'])
    max_length = st.number_input("Number of tokens to generate", 10, 1024, value=250)
    gen = st.form_submit_button('Generate')


if gen: st.text(generator(input, max_length=max_length, num_return_sequences=1)[0]['generated_text'])