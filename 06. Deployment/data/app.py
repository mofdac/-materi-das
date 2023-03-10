# Core Pkgs
import streamlit as st
import plotly.express as px
# sklearn version = 0.24.2

# EDA Pkgs
import pandas as pd
import numpy as np
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

# Utils
import joblib

# Image
from PIL import Image

st.sidebar.title("Analytics Web Application")
menu = ["EDA","Classification"]
choice = st.sidebar.selectbox("Select Menu", menu)
if choice == "EDA":
    st.title("Exploratory Data Analysis")
    st.sidebar.image('kode.jpg')
    data = st.file_uploader("Upload Dataset", type=["csv","txt"])
    if data is not None:
        df = pd.read_csv(data)
        st.dataframe(df.head())
    else:
        st.write("No Dataset To Show")
    st.subheader("Exploratory Data Analysis")
    if data is not None:
        if st.checkbox("Show Shape"):
            st.write(df.shape)
        if st.checkbox("Show Summary"):
            st.write(df.describe())
        if st.checkbox("Correlation Matrix"):
            st.write(sns.heatmap(df.corr(),annot=True))
            st.pyplot()

        all_columns = df.columns.to_list()
        if st.checkbox("Histogram"):
            columns_to_plot = st.selectbox("Select Column for Histogram", all_columns)
            hist_plot = df[columns_to_plot].plot.hist()
            st.write(hist_plot)
            st.pyplot()
        if st.checkbox("Pie Chart"):
            columns_to_plot = st.selectbox("Select Column for Pie Chart", all_columns)
            pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pie_plot)
            st.pyplot()
    st.subheader("Plotly Charts")
    if data is not None:
        plot_type = st.selectbox("Select Type of Plot",["bar","line","area","scatter","box"])
        selected_x = st.selectbox("Select X axis by Column", all_columns)
        selected_y = st.selectbox("Select Y axis by Column", all_columns)
        color_by = st.selectbox("Select Color by Column", all_columns)
        dfp = df
        dfp[color_by] = dfp[color_by].astype(str)
        if st.button("Generate Plot"):
            if plot_type == "bar":
                fig = px.bar(dfp,x=selected_x,y=selected_y,color = color_by)
                st.plotly_chart(fig)
            elif plot_type == "line":
                dfp = dfp.groupby(by=[selected_x,color_by],as_index=False)[selected_y].mean()
                fig = px.line(dfp,x=selected_x,y=selected_y,color = color_by)
                st.plotly_chart(fig)
            elif plot_type == "scatter":
                fig = px.scatter(dfp,x=selected_x,y=selected_y, color=color_by)
                st.plotly_chart(fig)
            elif plot_type == "box":
                fig = px.box(dfp, x=selected_x,y=selected_y)
                st.plotly_chart(fig)
            else:
                pass

elif choice == "Classification":
    st.subheader("Classification Prediction")
    iris= Image.open('iris.png')
    st.image(iris)

    modelfile = open("model.pkl", "rb")
    model = joblib.load(modelfile)

    st.sidebar.title("Features")
    #Intializing
    sl = st.sidebar.slider(label="Sepal Length (cm)",value=5.2,min_value=0.0, max_value=8.0, step=0.1)
    sw = st.sidebar.slider(label="Sepal Width (cm)",value=3.2,min_value=0.0, max_value=8.0, step=0.1)
    pl = st.sidebar.slider(label="Petal Length (cm)",value=4.2,min_value=0.0, max_value=8.0, step=0.1)
    pw = st.sidebar.slider(label="Petal Width (cm)",value=1.2,min_value=0.0, max_value=8.0, step=0.1)

    if st.button("Click Here to Classify"):
        dfvalues = pd.DataFrame(list(zip([sl],[sw],[pl],[pw])),columns =['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        input_variables = np.array(dfvalues[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
        prediction = model.predict(input_variables)
        if prediction == 1:
            st.image('setosa.png')
        elif prediction == 2:
            st.image('versicolor.png')
        elif prediction == 3:
            st.image('virginica.png')
