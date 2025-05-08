import streamlit as st
from src.blooms_process import process_syllabus_to_topics
from src.random_logic import generate_random_topics
from src.classify_topic import classify_topics_to_question_types
from src.create_questions import generate_questions_from_topics
import json

# Streamlit app configuration
st.set_page_config(page_title="Syllabus Question Generator", layout="wide")

# Title and description
st.title("Syllabus Question Generator")
st.write(
    "Upload a syllabus PDF (via S3 path), specify question counts, difficulties, "
    "and units to generate questions categorized by Bloom's Taxonomy."
)

# Input section
st.header("Input Parameters")
col1, col2 = st.columns(2)

# Column 1: S3 Path and Question Counts (interactive inputs)
with col1:
    s3_file_path = st.text_input(
        "S3 File Path",
        value="s3://anna-univ-qna/syllabus/CS23301.Syllabus.pdf",
        help="Enter the S3 URI of the syllabus PDF"
    )

    st.markdown("#### Question Counts per Part")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        part_a = st.number_input("MCQs", min_value=0, max_value=100, value=10, step=1)
    with col_b:
        part_b = st.number_input("Short Answer", min_value=0, max_value=100, value=6, step=1)
    with col_c:
        part_c = st.number_input("Long answer", min_value=0, max_value=100, value=4, step=1)

    # Combine into comma-separated format for compatibility
    question_counts = f"{part_a},{part_b},{part_c}"

# Column 2: Difficulty and Units
with col2:
    difficulties = st.multiselect(
        "Difficulty Levels",
        options=["easy", "medium", "hard"],
        default=["easy", "medium"],
        help="Select one or more difficulty levels"
    )

    units = st.multiselect(
        "Units",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        help="Select one or more units (1 to 5)"
    )

# Button to trigger processing
if st.button("Generate Questions"):
    try:
        # Step 1: Process syllabus to topics
        with st.spinner("Processing syllabus..."):
            topic_json = process_syllabus_to_topics(s3_file_path)
        st.subheader("Step 1: Extracted Topics (JSON)")
        st.json(topic_json)

        # Step 2: Generate random topics
        with st.spinner("Generating random topics..."):
            random_topics = generate_random_topics(
                question_counts=question_counts,
                units=units,
                difficulties=difficulties,
                syllabus_json=topic_json
            )
        st.subheader("Step 2: Randomly Selected Topics")
        st.text(random_topics)

        # Step 3: Classify topics into question types
        with st.spinner("Classifying topics..."):
            question_types = classify_topics_to_question_types(
                question_counts=question_counts,
                topic_list=random_topics
            )
        st.subheader("Step 3: Classified Topics with Question Types (JSON)")
        st.json(json.loads(question_types))

        # Step 4: Generate questions
        with st.spinner("Generating questions..."):
            questions = generate_questions_from_topics(topics_json=question_types)

        # Display in parts
        st.subheader("Step 4: Generated Questions")

        index = 0
        if part_a > 0:
            st.markdown("### Part A")
            for i in range(part_a):
                st.write(f"{i+1}. {questions[index]}")
                index += 1

        if part_b > 0:
            st.markdown("### Part B")
            for i in range(part_b):
                st.write(f"{i+1}. {questions[index]}")
                index += 1

        if part_c > 0:
            st.markdown("### Part C")
            for i in range(part_c):
                st.write(f"{i+1}. {questions[index]}")
                index += 1

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.write("Built with Streamlit and AWS services. Ensure AWS credentials are configured for S3 and Bedrock access.")
