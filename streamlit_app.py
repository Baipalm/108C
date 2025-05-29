import streamlit as st

st.set_page_config(
    page_title="NMF Demonstration",
)

st.write("Home")

st.sidebar.success("Select a demo above.")

st.markdown("[▶ Choose a Convergence](?page=nmf_convergence)")

st.markdown("[▶ Choose a Cost function ](?page=nmf_costfunctions)")
