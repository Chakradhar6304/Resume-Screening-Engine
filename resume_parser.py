# resume_parser.py

import spacy
import re
import json

# Load the spaCy model
# Make sure to run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    exit()

# Define keywords for skills (expand this list for better accuracy)
SKILL_KEYWORDS = [
    'python', 'java', 'c++', 'javascript', 'react', 'angular', 'vue', 'node.js',
    'django', 'flask', 'spring', 'ruby on rails', 'html', 'css', 'sql', 'nosql',
    'mongodb', 'postgresql', 'mysql', 'git', 'docker', 'kubernetes', 'aws',
    'azure', 'gcp', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
    'machine learning', 'deep learning', 'natural language processing', 'nlp',
    'data analysis', 'data visualization', 'project management', 'agile', 'scrum'
]

def extract_text_from_file(file_path):
    """
    Extracts text from a given file.
    Currently supports .txt files. Can be extended for .pdf, .docx, etc.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def extract_name(text):
    """
    Extracts the name from the resume text using spaCy's NER.
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            # Often the first PERSON entity is the candidate's name
            return ent.text
    return None

def extract_email(text):
    """
    Extracts email addresses from the text using regex.
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None

def extract_phone(text):
    """
    Extracts phone numbers from the text using regex.
    """
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    match = re.search(phone_pattern, text)
    return match.group(0) if match else None

def extract_skills(text):
    """
    Extracts skills from the text based on a predefined list of keywords.
    """
    found_skills = set()
    text_lower = text.lower()
    for skill in SKILL_KEYWORDS:
        if skill.lower() in text_lower:
            found_skills.add(skill)
    return list(found_skills)

def parse_resume(file_path):
    """
    Parses a resume file to extract text and structured information.
    """
    resume_text = extract_text_from_file(file_path)
    if not resume_text:
        return None, None

    # Use spaCy for more advanced parsing
    doc = nlp(resume_text)

    # Extract basic information
    name = extract_name(resume_text)
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    skills = extract_skills(resume_text)

    # Extract experience (a simplified approach)
    experience = []
    # This is a placeholder for a more sophisticated experience extraction logic.
    # A more robust solution would involve pattern matching for job titles, companies, and dates.
    # For now, we'll just pull sentences that might contain experience-related info.
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in ['experience', 'work', 'employment', 'project']):
            experience.append(sent.text.strip())

    # Create a structured profile
    profile = {
        'name': name,
        'contact': {
            'email': email,
            'phone': phone
        },
        'skills': skills,
        'experience_summary': experience
    }

    return resume_text, profile

if __name__ == '__main__':
    # This is for testing the parser directly
    test_file = 'resumes/sample_resume_se.txt'
    text, structured_profile = parse_resume(test_file)

    if text and structured_profile:
        print("--- Extracted Resume Text ---")
        print(text[:500] + "...") # Print first 500 chars
        print("\n--- Structured Profile ---")
        print(json.dumps(structured_profile, indent=2))