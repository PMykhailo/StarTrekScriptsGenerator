import streamlit as st
import pandas as pd

st.title('This is GPT-2 model fine-tuned with Star Trek scipts')
st.markdown('''Write a line for  model to begin  with and it will try it's best to continue it as Star Trek script. Please write caracter name in CAPS.''')

if 'tokenizer' not in st.session_state:
    from transformers import GPT2Tokenizer
    st.session_state['tokenizer'] = GPT2Tokenizer.from_pretrained('gpt2')
if 'model' not in st.session_state:
    from transformers import pipeline, set_seed
    from transformers import GPT2LMHeadModel
    st.session_state['model'] = GPT2LMHeadModel.from_pretrained(r'''F:\files\ML\PhrasePrediction\models\all_scripts''')
    st.session_state['generator'] = pipeline('text-generation', model=st.session_state['model'], tokenizer=st.session_state['tokenizer'], device=st.session_state['model'].device)
if 'al' not in st.session_state:
    st.session_state['al'] = pd.read_csv(r'F:\files\ML\PhrasePrediction\STscripts\all_lines')

def gen(generator,inp,max_length):
    st.session_state['output'] = generator(inp, max_length=max_length, num_return_sequences=1)[0]['generated_text']

#set_seed(42)
random_line = st.button('Start with random line from database')
if random_line:
    st.session_state['input_val'] = st.session_state['al'].sample(n=1)['0'].item()
else:
    st.session_state['input_val'] = '''PICARD: You will agree, Data, that Starfleet's orders are difficult?'''
with st.form(key = 'parametres'):
    input = st.text_input('Enter a starting line here',value=st.session_state['input_val'])
    max_length = st.number_input("Number of tokens to generate", 250, 1250)
    gen = st.form_submit_button('Generate', on_click=gen, args=(st.session_state['generator'], input, max_length))
#start = st.button('Generate')
if 'output' in st.session_state:
    st.text(st.session_state['output'])