To install dependencies:

    pip install -r requirements.txt

To run a code:
    
    streamlit run [folder]/[filename]
    i.e., 
        streamlit run supervised_learning/Naive_Bayes.py

A way to debug your Streamlit application in Pycharm:

    Edit run/debug configuration
    Change script path t-> Module name
    Fill Module Name: streamlit.cli
    Fill Paramateres: run <full_path_filename>
