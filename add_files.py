# Before running this script, you must install the required libraries:
# pip install opensearch-py requests-aws4auth boto3 langchain langchain-aws pypdf2 python-dotenv

import os
import boto3
from dotenv import load_dotenv
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from a .env file if it exists.
# This is useful for storing your AWS credentials securely.
load_dotenv()

# --- CONFIGURATION ---
# Your OpenSearch domain endpoint from the conversation.
OPENSEARCH_URL = "https://search-test1-annauniv-pcx3f52wxykhpd4md6v4bjeqdy.ap-south-1.es.amazonaws.com"
# The name of the index you want to create or use.
INDEX_NAME = "test-annauniv"
# The AWS region where your OpenSearch domain is located.
REGION = "ap-south-1"
# The service name for signing requests.
SERVICE = "es"

# --- AWS AUTHENTICATION ---
# This code uses Boto3 to automatically get your credentials from
# your environment (e.g., AWS CLI, environment variables).
print("Attempting to load AWS credentials from environment...")
try:
    session = boto3.Session()
    credentials = session.get_credentials()
    if not credentials:
        raise ValueError("AWS credentials not found. Please configure your AWS CLI or environment variables.")
    
    # Get the credentials, which will be used to sign the requests.
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        REGION,
        SERVICE,
        session_token=credentials.token
    )
    print("Successfully loaded AWS credentials.")
except Exception as e:
    print(f"Authentication Error: {e}")
    exit()

# --- HELPER FUNCTION TO INGEST DATA ---
def ingest_pdf_to_vectordb(file_path):
    """
    Loads a PDF document, splits it into chunks, and ingests it into an OpenSearch vector database.
    
    Args:
        file_path (str): The path to the PDF file to be processed.
    """
    print(f"\nProcessing file: {file_path}")
    
    # 1. Load the document using Amazon Textract.
    # Note: Ensure you have permissions for Amazon Textract.
    # The file path should be a full S3 URI.
    loader = AmazonTextractPDFLoader(file_path=file_path)
    try:
        documents = loader.load()
        if not documents:
            print("Warning: No documents were loaded from the file.")
            return
        print(f"Document loaded successfully. Found {len(documents)} pages.")
    except Exception as e:
        print(f"Error loading document with Amazon Textract: {e}")
        return

    # 2. Split the documents into smaller chunks for embedding.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    print(f"Document split into {len(docs)} chunks.")

    # 3. Create a BedrockEmbeddings instance.
    print("Creating Bedrock Embeddings model...")
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

    # 4. Ingest the chunks into the OpenSearch vector database.
    print("Ingesting document chunks into OpenSearch VectorSearch...")
    try:
        docsearch = OpenSearchVectorSearch(
            opensearch_url=OPENSEARCH_URL,
            index_name=INDEX_NAME,
            embedding_function=embeddings,
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            bulk_size=2500
        )
        docsearch.add_documents(docs)
        print("Documents successfully added to the index.")
    except Exception as e:
        print(f"Error ingesting documents to OpenSearch: {e}")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # The actual path to your PDF file in the S3 bucket.
    my_file_path = "s3://anna-univ-qna/Textbooks/CS23301.T1.pdf"

    # Call the ingestion function.
    ingest_pdf_to_vectordb(my_file_path)

    print("\nIngestion process completed.")
