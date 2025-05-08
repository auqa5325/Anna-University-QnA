from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import json

def generate_questions_from_topics(topics_json: str) -> list:
    """
    Retrieve relevant documents and generate questions for each topic based on its weightage.

    Args:
        topics_json (str): JSON string containing topics and their question types
                          (e.g., [{"topic": "Unit 1 - Topic (Level)", "question_type": "Very Short Answer"}, ...])

    Returns:
        list: List of generated questions, one per topic

    Raises:
        ValueError: If topics_json is invalid
        Exception: If document retrieval or question generation fails
    """
    # Parse topics JSON
    try:
        topics = json.loads(topics_json)
        if not topics or not isinstance(topics, list):
            raise ValueError("Topics JSON must be a non-empty list")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for topics")

    # AWS credentials and region
    region = 'ap-south-1'
    service = 'es'
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

    # OpenSearch domain endpoint
    host = 'search-anna-univ-ivzhdnmtdsvf2wvtbmzsmfrmym.ap-south-1.es.amazonaws.com'

    # Initialize OpenSearch vector store
    vector_store = OpenSearchVectorSearch(
        opensearch_url=f"https://{host}",
        index_name="test-annauniv",
        embedding_function=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0"),
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    # Initialize LLM
    llm = ChatBedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        model_kwargs={"temperature": 0, "max_gen_len": 1024}
    )

    # Define prompt template
    prompt_template = PromptTemplate(
    input_variables=["query", "document"],
    template="""
You are a precise academic question generator. Only return questions in the specified format with no extra text. Follow these instructions exactly:

Input:
- Topic, type, and difficulty: {query} (format: "Topic | Type | Difficulty", e.g., "Software Development | MCQ | Medium")
- Reference material: {document} (course content, e.g., book chapter or syllabus excerpt)

Instructions:
1. Generate ONE question based on the topic, question type (MCQ, Short Answer, or Long Answer), and difficulty level.
2. **Do NOT start the question with "What".**
3. Instead, vary the question starters using words like: "If", "Why", "Mention", "Compare", "Define", "Explain", "Discuss", "How", "Suggest", "Provide", "Analyze", etc.
4. Match the question's complexity to the difficulty level:
   - Easy: Recall or definition (e.g., "Define Quality Assurance.")
   - Medium: Application or explanation (e.g., "Explain the concept of Control structure testing.")
   - Hard: Analysis or synthesis (e.g., "Analyze the trade-offs between process scheduling algorithms.")
5. Format the question based on type:
   - For MCQ:
     - Write a clear question stem.
     - Provide exactly four distinct, plausible options labeled A), B), C), D).
     - Ensure one correct answer, with distractors closely related to match the difficulty level.
   - For Short Answer:
     - Write a concise question (20-30 words) requiring a brief response.
   - For Long Answer:
     - Write a focused question (30-50 words) requiring detailed explanation or analysis.
6. Base the question strictly on the reference material for relevance and accuracy, ensuring alignment with academic Software Engineering concepts.
7. Ensure the question is clear, unique, and suitable for an academic paper.
8. Output ONLY the question (with options for MCQ). No explanations, labels, or extra text.
"""
)

    # Create runnable sequence
    chain = RunnableSequence(prompt_template | llm)

    # Generate questions for each topic
    questions = []
    for topic_entry in topics:
        try:
            topic = topic_entry.get("topic")
            question_type = topic_entry.get("question_type")
            if not topic or not question_type:
                raise ValueError("Each topic entry must have 'topic' and 'question_type'")

            # Form query for document retrieval
            query = f"{topic} question_type: {question_type}"

            # Retrieve relevant documents
            results = vector_store.max_marginal_relevance_search(
                query=query,
                k=5,           # Number of documents to return
                fetch_k=20,    # Number of documents to fetch before applying MMR
                lambda_mult=0.5  # Balance between relevance and diversity
            )

            # Combine documents into a single string
            final_docs = "".join([doc.page_content for doc in results]).replace("\n", " ").replace("\r", " ")

            # Generate question
            inputs = {
                "query": query,
                "document": final_docs
            }
            result = chain.invoke(inputs)
            question = result.content.strip()
            questions.append(question)
        except Exception as e:
            questions.append(f"Error generating question for {topic}: {str(e)}")

    return questions