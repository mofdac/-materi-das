import streamlit as st 
import plotly.express as px 
import matplotlib.pyplot as plt
import pandas as pd 
import requests
from st_aggrid import AgGrid
#baca dataframe dari file csv 
titanic = pd.read_csv('https://raw.githubusercontent.com/mofdac/-materi-das/main/01.%20Python%20for%20DA/titanic.csv')


def main() : 
    #matplotlib chart 
    fig,ax = plt.subplots()
    plt.scatter(titanic['Age'],titanic['Fare'])
    st.pyplot(fig)
    plotly_fig = px.scatter(titanic['Age'],titanic['Fare'])
    st.plotly_chart(plotly_fig)
if __name__ == '__main__' : 
    main()
