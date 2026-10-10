import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Food Recommendation System", page_icon="🍔", layout="wide"
)

st.title("🍔 Food Recommendation System")
st.write("Food Recommendation System using Apriori Algorithm and Association Rules")

rules = pd.read_csv("association_rules.csv")
baskets = pd.read_csv("processed_baskets.csv")

st.success("Dataset and association rules loaded successfully!")

st.metric("Total Transactions", len(baskets))
st.metric("Total Association Rules", len(rules))
