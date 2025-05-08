from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import json
import re

def classify_topics_to_question_types(question_counts: str, topic_list: str) -> str:
    """
    Classify topics into question types based on question counts and Bloom's Taxonomy.

    Args:
        question_counts (str): Comma-separated string of question counts (e.g., '10,6,4' or '5,5')
        topic_list (str): String of randomly selected topics, one per line, numbered

    Returns:
        str: JSON string with topics and their assigned question types

    Raises:
        ValueError: If question counts are invalid
        Exception: If LLM processing or JSON parsing fails
    """
    # Parse question counts
    try:
        counts = [int(x) for x in question_counts.split(',')]
        if not counts:
            raise ValueError("Question counts cannot be empty")
        total_questions = sum(counts)
    except ValueError:
        raise ValueError("Invalid question counts format. Use comma-separated integers (e.g., '10,6,4')")

    # Determine question types based on number of inputs
    if len(counts) == 3:
        question_types = ["MCQs", "Short Answer", "Long Answer"]
    else:
        question_types = [f"{marks} Marks" for marks in counts]  # Fallback to mark-based naming

    # Create question_counts dictionary
    question_counts_dict = {q_type: count for q_type, count in zip(question_types, counts)}

    # Process topic_list to extract topics
    topics = []
    for line in topic_list.split('\n'):
        if line.strip() and not line.startswith("Selected Topics:"):
            topic = line.split('.', 1)[1].strip() if '.' in line else line.strip()
            topics.append(topic)

    # Initialize LLM
    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="ap-south-1",
        model_kwargs={"temperature": 0.1, "max_tokens": 2048}
    )

    # Define prompt template
    prompt_template = PromptTemplate(
        input_variables=["topic_list", "question_counts"],
        template="""
You are an expert exam paper designer.

Given:
1. A list of selected topics, each with any existing labels.
2. The required number of questions in each category:
   {question_counts}

Your task:
- Assign each topic to one of these question formats:
  • MCQ  
  • Short Answer  
  • Long Answer  
- Convert any topic that would yield a “Very Short Answer” into an MCQ.
- Use MCQs for topics that test factual recall or discrete concepts.
- Use Short Answer for topics requiring concise explanations or definitions.
- Use Long Answer for topics needing detailed reasoning, analysis, or examples.
- You do not need to consider marks or difficulty levels—tough questions may be MCQs, and simple questions may be Long Answer.
- Exactly match the total counts specified in {question_counts} (e.g., “10,6,4” means 10 MCQs, 6 Short Answers, 4 Long Answers).
- **Output ordering requirement:** List all MCQs first, then all Short Answer questions, then all Long Answer questions in the returned JSON.

Return only JSON in this format, respecting the above order:
[
  {{
    "topic": "Topic Name(difficulty)",
    "question_type": "MCQ" | "Short Answer" | "Long Answer"
  }},
  ...
]

Topics:
{topic_list}
"""
    )

    # Create runnable sequence
    chain = RunnableSequence(prompt_template | llm)

    # Invoke chain
    try:
        result = chain.invoke({
            "topic_list": "\n".join(topics),
            "question_counts": json.dumps(question_counts_dict)
        })

        # Preprocess result to remove markdown-style code blocks
        cleaned_output = result.content.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json"):].strip()
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3].strip()

        # Fallback: Extract JSON array using regex if needed
        try:
            output_json = json.loads(cleaned_output)
        except json.JSONDecodeError:
            match = re.search(r'\[\s*{.*}\s*]', cleaned_output, re.DOTALL)
            if not match:
                raise Exception("No valid JSON array found in LLM output.")
            output_json = json.loads(match.group(0))

        return json.dumps(output_json, indent=2)

    except Exception as e:
        raise Exception(f"Error classifying topics: {str(e)}")
