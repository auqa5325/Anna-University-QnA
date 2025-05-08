from langchain.document_loaders import AmazonTextractPDFLoader
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws.embeddings import BedrockEmbeddings
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

# Define AWS Auth for OpenSearch (Make sure boto3 has correct credentials configured)
region = "ap-south-1"
service = "es"

credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    service,
    session_token=credentials.token
)


def file_to_vectordb(file_path):
    loader = AmazonTextractPDFLoader(file_path=file_path)

    try:
        documents = loader.load()
        print("Document loaded successfully.")
    except Exception as e:
        print(f"Error loading document: {e}")
        raise

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    docsearch = OpenSearchVectorSearch(
    opensearch_url="https://search-anna-univ-ivzhdnmtdsvf2wvtbmzsmfrmym.ap-south-1.es.amazonaws.com",
    index_name="test-annauniv",
    embedding_function=embeddings,
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    bulk_size=2500,
    connection_class=RequestsHttpConnection
)
    docsearch.add_documents(docs)
    print("Documents added to existing index.")


