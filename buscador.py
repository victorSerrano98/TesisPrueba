import spacy
import requests
import torch
from transformers import pipeline
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertForQuestionAnswering, BertConfig
from rank_bm25 import BM25Okapi
import pandas as pd
import spacy
import requests


tokenizer = AutoTokenizer.from_pretrained("graviraja/covidbert_squad")
model = AutoModelForQuestionAnswering.from_pretrained("graviraja/covidbert_squad")



max_length = 512 # The maximum length of a feature (question and context)

#question_answerer = pipeline("question-answering")
nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)
nlpPipe = pipeline("question-answering")
# question = "¿Cuánto tiempo tardan en aparecer los síntomas del covid-19?"
def traductor(question):
    translator = Translator()
    translation = translator.translate(question, dest='en')
    question = translation.text
    return question

def spa(question):
    nlpSp = spacy.load("en_core_web_sm")
    preguntaFormt = " ".join(question.split())
    print(preguntaFormt)
    doc = nlpSp(preguntaFormt)
    a = [token.dep_ for token in doc]
    print("Estructura Frase: ", a)

    # Create list of word tokens
    token_list = []
    for token in doc:
        token_list.append(token.text)
    # Create list of word tokens after removing stopwords
    filtered_sentence = []

    for word in token_list:
        lexeme = nlpSp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word)

    query = filtered_sentence

    my_stopwords = ['long', '?', '¿', '!', 'covid-19', 'covid', 'coronavirus', 'COVID-19', 'COVID19', 'covid19' 'SARS-COV2', 'SARS-COV 2', 'sarscov2', 'sars cov 2']
    tokens_filtered = [word for word in query if not word in my_stopwords]
    #print("111: ", tokens_filtered)
    query = tokens_filtered
    var = ""
    for i in query:
        var = var + " " + "\"" + i + "\""
        #print(var)

    var = var + " \"covid-19\""
    var = " ".join(var.split())
    print("formateada: ", var)
    return var

def consultaAPI(query, num):
    data = {
        "query": query,
        "offset": 0,
        "limit": 100,
        "fields": "title,abstract,url,year"

    }
    r = requests.get('https://api.semanticscholar.org/graph/v1/paper/search', params=data).json()
    # print("r::::", r)
    try:
        if r['message']:
            print("Heyyyyyyy")
            return None
    except:
        output_dict = [x for x in r["data"] if x['year'] == 2021]
        return r

def sp_Abstract(abstract):
    nlpAbst = spacy.load("en_core_web_sm")
    preguntaFormt = " ".join(abstract.split())
    query = ""
    print(preguntaFormt)
    doc = nlpAbst(preguntaFormt)
    a = [token.dep_ for token in doc]

    # Create list of word tokens
    token_list = []
    for token in doc:
        token_list.append(token.text)
    # Create list of word tokens after removing stopwords
    filtered_sentence = []

    for word in token_list:
        lexeme = nlpAbst.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word)

    query = filtered_sentence

    my_stopwords = ['?', '¿', '!', '\n']
    tokens_filtered = [word for word in query if not word in my_stopwords]
    query = tokens_filtered
    var = ""
    for i in query:
        var = var + " " + i + " "

    return var


def respuesta(question,text):
    list = ""
    cont = 0
    contArtc = 0
    contArtcTotal = 0
    for res in text:
        contArtcTotal = contArtcTotal + 1

        try:
            abstracCrrg = res["abstract"]
            result = nlp(question=question, context=abstracCrrg)
            rest = nlpPipe(question=question, context=abstracCrrg)
            x = (round(rest['score'], 4))

            # TOKENIZADOR DE LA PREGUNTA PARA OBTENER RESPUESTA ACORDE
            inputs = tokenizer.encode_plus(question, abstracCrrg,
                                           add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                           truncation=True,
                                           max_length=512,
                                           return_tensors='pt')  # Return pytorch tensors.
            #nlp({'question': question, 'context': res["abstract"]})
            # print(res["title"])
            if x > 0.00:
                cont = cont + 1

                titulo = res["title"]

                input_ids = inputs["input_ids"].tolist()[0]
                text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                answer_start_scores, answer_end_scores = model(**inputs, return_dict=False)
                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1
                answer = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
                translator = Translator()
                if answer:
                    if '[CLS]' in answer:
                        translation = translator.translate(result['answer'], dest='es')
                    else:
                        translation = translator.translate(answer, dest='es')
                    respTraducida = translation.text
                else:
                    translation = translator.translate(result['answer'], dest='es')
                    respTraducida = translation.text
                list =  list + "Respuesta: " + respTraducida + "\n \n Fuente: " + titulo + "\n \n" + "Score: " + str(x) + "\n \n"


        except:
            print("Error")
        if contArtc == 1:
            contArtc = 0
            if cont > 20:
                break
        contArtc = contArtc + 1
    print("Art. Totales: ", contArtcTotal)
    return list, cont


def busqueda(query, text):
    corpus = text
    tokenized_corpus = [doc['abstract'].split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    #query = "symptoms"
    tokenized_query = query.lower().split(" ")
    print(tokenized_query)
    resultados = bm25.get_top_n(tokenized_query, corpus, n=15)
    return resultados

