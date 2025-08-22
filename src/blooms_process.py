import re
import json
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

def process_syllabus_to_topics(s3_file_path: str) -> dict:
    """
    Process a syllabus PDF from an S3 path and extract topics categorized by Bloom's Taxonomy.
    
    Args:
        s3_file_path (str): S3 URI of the syllabus PDF (e.g., 's3://bucket-name/path/to/file.pdf')
    
    Returns:
        dict: JSON object with topics categorized by Bloom's Taxonomy for each unit
    
    Raises:
        Exception: If PDF loading, processing, or JSON parsing fails
    """
    # Preprocess text function
    def preprocess_syllabus_text(syllabus_text):
        text = syllabus_text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'^\s*[\d\.\-\*]+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'(page\s*\d+|syllabus|course\s*code|semester|department\s*of\s*[a-z\s]+)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text

    try:
        # Initialize PDF loader
        loader = AmazonTextractPDFLoader(file_path=s3_file_path,region_name="ap-south-1")
        
        # Load and extract text
        documents = loader.load()
        syllabus_text = "".join([doc.page_content for doc in documents])
        
        # Preprocess text
        processed_text = preprocess_syllabus_text(syllabus_text)
        
        # Initialize LLM
        llm = ChatBedrock(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region_name="ap-south-1",
            model_kwargs={"max_tokens": 8192}
        )
        
        # Define prompt template
        prompt_template = PromptTemplate(
            input_variables=["syllabus_text"],
            template="""
You are a curriculum analysis assistant.

Given the syllabus text below, extract topic **titles only** (no descriptions, no actions, no rephrasing) for each of the 5 units. Categorize them under Bloom's Taxonomy levels:

- Remembering
- Understanding
- Applying
- Analyzing
- Evaluating
- Creating

**Rules**:
- Use exact topic wording as it appears in the syllabus.
- Do NOT generate explanations, descriptions, or action phrases.
- Each level MUST contain at least one topic.
- Topics can repeat across levels if applicable.
- Return only valid, compact JSON — no formatting, no extra commentary, no Markdown.

Syllabus Text:
{syllabus_text}

Output format:
{{
  "units": [
    {{
      "unit_number": 1,
      "topics": {{
        "Remembering": ["..."],
        "Understanding": ["..."],
        "Applying": ["..."],
        "Analyzing": ["..."],
        "Evaluating": ["..."],
        "Creating": ["..."]
      }}
    }},
    {{
      "unit_number": 2,
      "topics": {{
        "Remembering": ["..."],
        "Understanding": ["..."],
        "Applying": ["..."],
        "Analyzing": ["..."],
        "Evaluating": ["..."],
        "Creating": ["..."]
      }}
    }},
    {{
      "unit_number": 3,
      "topics": {{
        "Remembering": ["..."],
        "Understanding": ["..."],
        "Applying": ["..."],
        "Analyzing": ["..."],
        "Evaluating": ["..."],
        "Creating": ["..."]
      }}
    }},
    {{
      "unit_number": 4,
      "topics": {{
        "Remembering": ["..."],
        "Understanding": ["..."],
        "Applying": ["..."],
        "Analyzing": ["..."],
        "Evaluating": ["..."],
        "Creating": ["..."]
      }}
    }},
    {{
      "unit_number": 5,
      "topics": {{
        "Remembering": ["..."],
        "Understanding": ["..."],
        "Applying": ["..."],
        "Analyzing": ["..."],
        "Evaluating": ["..."],
        "Creating": ["..."]
      }}
    }}
  ]
}}
"""
        )
        
        # Create runnable sequence
        chain = RunnableSequence(prompt_template | llm)
        
        # Invoke chain to get categorized topics
        result = chain.invoke({"syllabus_text": processed_text})
        
        # Parse result to JSON
        output_json = json.loads(result.content)
        return output_json
    
    except json.JSONDecodeError:
        raise Exception("Failed to parse LLM output as JSON. Raw output: " + result.content)
    except Exception as e:
        raise Exception(f"Error processing syllabus: {str(e)}")