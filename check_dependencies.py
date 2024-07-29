
# Run the pip command to install dependencies:
# pip install -qU langchain==0.0.354 openai==1.6.1 datasets==2.10.1 pinecone-client==3.0.0 tiktoken==0.5.2
# pip install scikit-learn gensim
# pip install -U langchain-openai


'''Run these dependencies in single line
pip install -qU \
    langchain==0.0.354 \
    openai==1.6.1 \
    datasets==2.10.1 \
    pinecone-client==3.0.0 \
    tiktoken==0.5.2'''

# Python code to verify the installation of the packages
try:
    import langchain, openai, datasets, pinecone, tiktoken
    print("All packages are installed successfully.")
except ImportError as e:
    print(f"An error occurred: {e}")

# Add further Python code below
