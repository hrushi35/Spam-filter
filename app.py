import streamlit as st
import joblib

text_model = joblib.load("spam fighter")
vectorizer = joblib.load("Vectorizer")

st.title("SPAM FILTER")

ip = st.text_input("Enter the message :")  
   
if st.button('PREDICT'):
    if ip.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Vectorize the input
        ip_vec = vectorizer.transform([ip])
        
        # Predict
        op = text_model.predict(ip_vec)
        
        # Show result
        st.title(op[0])                            
