import time
from time import sleep
import streamlit
import streamlit as st
import pandas as pd

import buscador

### Encabezado ###
st.set_page_config(page_title="Chatbot Covid-19")
st.markdown('___')
st.markdown("<h4 style='text-align: center; color:grey;'>Universidad Nacional de Loja</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color:grey;'>Ingeniería en Sistemas</h4>", unsafe_allow_html=True)
st.markdown(
    "<h1 style='text-align: center; color:Black;frameborder=1;'>""Agente virtual para brindar asistencia acerca del Covid-19""</h1>",
    unsafe_allow_html=True)
# ![Star (https://unl.edu.ec/sites/default/files/inline-images/logo_0.png?style=social)](https://unl.edu.ec/)

st.markdown('___')


######

def classify(num):
    if num == 0:
        return 'setosa'
    else:
        return 'virginica'


def main():
    # st.title('Agente Virtual (Covid-19):')
    # st.sidebar.header('Universidad Nacional de Loja')

    def parametros():
        pregunta = st.text_input('Ingrese su pregunta sobre el Covid-19:')

        return pregunta

    consultaAvanz = st.checkbox('Consulta Avanzada')

    pregunta = parametros()
    if st.button('Enviar pregunta'):
        import time
        t0 = time.time()
        traducido = buscador.traductor(pregunta)
        print(traducido)
        query = buscador.spa(traducido)
        print(query)
        print("*********************")
        consulta = buscador.consultaAPI(query, 1)
        print(consulta)
        output_dict = [x for x in consulta["data"] if x['abstract'] != None]
        txt = buscador.busqueda(query, output_dict)
        print(len(txt))
        resp = buscador.respuesta(traducido, txt)
        print("Respuestas \n \n ")

        if len(resp) < 1:
            streamlit.error("No se encontraron respuestas relacionadas." + "\n\n Reformula tu pregunta.")
            streamlit.error('Si no obtuvo una respuesta acorde, pruebe la \"Consulta Avanzada\"')

        for i in resp:
            print(i)
            if len(resp) > 1:
                st.success(i)
        t1 = time.time()
        print(f'Searched records in {round(t1 - t0, 3)} seconds \n')



    st.markdown('___')

if __name__ == '__main__':
    main()
