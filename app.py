import streamlit as st

st.title("Advance Global Warming Analysis")

# Upload the file
uploaded_file = st.file_uploader("Upload Python Script", type=["py"])

if uploaded_file is not None:
    # Save the uploaded file
    with open("uploaded_script.py", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully! You can now execute it.")
    
    # Execute the uploaded script
    exec(open("uploaded_script.py").read())
