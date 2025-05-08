import random
from typing import List, Dict

def generate_random_topics(question_counts: str, units: List[int], difficulties: List[str], syllabus_json: Dict) -> str:
    """
    Generate random topics based on question counts, units, difficulties, and syllabus JSON.
    
    Args:
        question_counts (str): Comma-separated string of question counts for parts (e.g., '10,6,4' or '5,5')
        units (List[int]): List of unit numbers (1 to 5)
        difficulties (List[str]): List of difficulty levels ('easy', 'medium', 'hard')
        syllabus_json (Dict): JSON output from process_syllabus_to_topics with categorized topics
    
    Returns:
        str: String of selected topics, one per line, numbered
    
    Raises:
        ValueError: If inputs are invalid (question counts, units, difficulties)
    """
    # Parse question counts
    try:
        parts = [int(x) for x in question_counts.split(',')]
        if not parts:
            raise ValueError("Question counts cannot be empty")
        num_questions = sum(parts)
    except ValueError:
        raise ValueError("Invalid question counts format. Use comma-separated integers (e.g., '10,6,4' or '5,5')")

    # Validate inputs
    if num_questions <= 0:
        raise ValueError("Total number of questions must be positive")
    if not difficulties or not all(d in ['easy', 'medium', 'hard'] for d in difficulties):
        raise ValueError("Difficulty levels must be 'easy', 'medium', or 'hard'")
    if not units or not all(1 <= u <= 5 for u in units):
        raise ValueError("Units must be between 1 and 5")

    # Bloom's hierarchy and mapping functions
    def get_blooms_hierarchy():
        return ['Remembering', 'Understanding', 'Applying', 'Analyzing', 'Evaluating', 'Creating']

    def map_difficulty_to_blooms(difficulties: List[str]) -> List[str]:
        blooms_map = {
            'easy': ['Remembering', 'Understanding'],
            'medium': ['Applying', 'Analyzing'],
            'hard': ['Evaluating', 'Creating']
        }
        blooms_levels = []
        for difficulty in difficulties:
            if difficulty not in blooms_map:
                raise ValueError(f"Invalid difficulty level: {difficulty}")
            blooms_levels.extend(blooms_map[difficulty])
        return list(set(blooms_levels))

    def fallback_blooms(current_blooms: List[str]) -> List[str]:
        full_hierarchy = get_blooms_hierarchy()
        current_indices = [full_hierarchy.index(b) for b in current_blooms if b in full_hierarchy]
        min_index = min(current_indices) if current_indices else 0
        return full_hierarchy[:min_index]

    # Select random topics
    primary_blooms = map_difficulty_to_blooms(difficulties)
    used_topics = set()
    selected_topics = []

    def collect_topics(blooms_levels: List[str]) -> List[str]:
        topics_found = []
        for unit in syllabus_json['units']:
            if unit['unit_number'] in units:
                for level in blooms_levels:
                    for topic in unit['topics'].get(level, []):
                        key = f"Unit {unit['unit_number']} - {topic} ({level})"
                        if key not in used_topics:
                            topics_found.append(key)
        return topics_found

    topics = collect_topics(primary_blooms)
    random.shuffle(topics)
    selected_topics.extend(topics[:num_questions])
    used_topics.update(selected_topics)

    while len(selected_topics) < num_questions:
        fallback_levels = fallback_blooms(primary_blooms)
        if not fallback_levels:
            break
        topics = collect_topics(fallback_levels)
        random.shuffle(topics)
        for t in topics:
            if len(selected_topics) >= num_questions:
                break
            if t not in used_topics:
                selected_topics.append(t)
                used_topics.add(t)
        primary_blooms = fallback_levels

    if not selected_topics:
        raise ValueError("No topics available even after fallback")

    # Format output as string
    output = ["Selected Topics:"]
    for i, topic in enumerate(selected_topics[:num_questions], 1):
        output.append(f"{i}. {topic}")
    return "\n".join(output)