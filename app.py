import streamlit as st
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os
import pickle
import io
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
import re
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document
import json

# Define the scopes your app will need
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Job roles and skills for filtering candidates
JOB_ROLES = {
    'Software Engineer': {
        'skills': ['Python', 'Java', 'C++', 'Data Structures', 'Algorithms', 'OOP', 'Git', 'SQL', 'HTML', 'CSS', 'JavaScript', 'System Design', 'Cloud Technologies', 'Docker', 'Kubernetes'],
        'projects': ['Web Development', 'System Design', 'Microservices', 'Cloud Computing', 'DevOps'],
    },
    'Data Scientist': {
        'skills': ['Python', 'Machine Learning', 'Statistics', 'Data Visualization', 'Pandas', 'NumPy', 'Deep Learning', 'NLP', 'Big Data'],
        'projects': ['Data Analysis', 'Predictive Modeling', 'Natural Language Processing', 'Data Pipelines'],
    },
    'Machine Learning Engineer': {
        'skills': ['Python', 'Deep Learning', 'Neural Networks', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Model Deployment', 'Hyperparameter Tuning'],
        'projects': ['Deep Learning Models', 'Reinforcement Learning', 'Model Deployment'],
    },
    # Add more job roles and their skills here as needed
}

def analyze_resume_with_openai(text, job_role):
    prompt = (
        f"Analyze the following resume text and evaluate the candidate for the job role: {job_role}.\n"
        f"Extract the following details:\n"
        f"1. Candidate's Name\n"
        f"2. Relevant Skills\n"
        f"3. Relevant Projects\n"
        f"4. Experience Score (0-10)\n"
        f"5. ATS Score for the job role\n\n"
        f"Resume Text: {text}\n\n"
        f"Format your response as JSON:\n"
        f'{{\n  "name": "Name",\n  "skills": ["Skill1", "Skill2"],\n'
        f'  "projects": [\n    {{\n      "project_name": "Project Name",\n'
        f'      "technologies": ["Technology1", "Technology2"],\n'
        f'      "description": "Brief Description"\n    }}\n  ],\n'
        f'  "experience_score": "Score",\n  "ats_score": "Score"\n}}'
    )

    messages = [
        {"role": "system", "content": "You are a resume analysis assistant."},
        {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-4",  # Specify the model
        messages=messages,
        max_tokens=1024,
    )

    final_res = response.choices[0].message.content
    return final_res


def authenticate_google_drive():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)


def get_files_from_drive(drive_service, folder_id):
    query = f"'{folder_id}' in parents"
    results = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return results.get('files', [])


def download_file(drive_service, file_id, file_name):
    fh = io.BytesIO()
    try:
        request = drive_service.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return fh
    except Exception as e:
        st.error(f"Error downloading file: {e}")
        return None


def parse_resume(file_data, file_name):
    file_extension = file_name.split('.')[-1].lower()
    
    if file_extension == "pdf":
        reader = PdfReader(file_data)
        return "".join(page.extract_text() for page in reader.pages)
    
    elif file_extension == "docx":
        doc = Document(file_data)
        return "\n".join(para.text for para in doc.paragraphs)
    
    else:
        st.error("Unsupported file type.")
        return None


def filter_candidate_based_on_skills_and_projects(analysis_json, job_role):
    skills = analysis_json.get('skills', [])
    projects = analysis_json.get('projects', [])

    required_skills = JOB_ROLES.get(job_role, {}).get('skills', [])
    required_projects = JOB_ROLES.get(job_role, {}).get('projects', [])

    # Check if candidate has the required skills and projects
    skill_match = len(set(skills) & set(required_skills)) > 0
    project_match = any(project['project_name'] in required_projects for project in projects)

    if skill_match and project_match:
        return True
    return False


def main():
    st.title("Resume Analysis Tool")

    folder_url = st.text_input("Enter Google Drive Folder URL:")
    job_role = st.selectbox("Select the Job Role", list(JOB_ROLES.keys()))

    if folder_url and job_role:
        folder_id = re.search(r'\/folders\/([a-zA-Z0-9_-]+)', folder_url)
        if folder_id:
            drive_service = authenticate_google_drive()
            files = get_files_from_drive(drive_service, folder_id.group(1))
            document_names = [file['name'] for file in files if file['mimeType'] in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]]
            
            if document_names:
                selected_file = st.selectbox("Select a document to process", document_names)
                if st.button("Analyze Resume"):
                    selected_file_data = next(file for file in files if file['name'] == selected_file)
                    file_data = download_file(drive_service, selected_file_data['id'], selected_file_data['name'])
                    if file_data:
                        resume_text = parse_resume(file_data, selected_file_data['name'])
                        if resume_text:
                            # Analyze the resume
                            analysis = analyze_resume_with_openai(resume_text, job_role)

                            if analysis:
                                try:
                                    analysis_json = json.loads(analysis)

                                    # Display the analysis
                                    st.write("Resume Analysis:")
                                    st.write(f"Name: {analysis_json.get('name', 'Not Found')}")
                                    st.write(f"Skills: {', '.join(analysis_json.get('skills', []))}")
                                    st.write("Projects:")
                                    for project in analysis_json.get('projects', []):
                                        st.write(f"- {project.get('project_name', 'Not Found')}")
                                    st.write(f"Experience Score: {analysis_json.get('experience_score', 'Not Found')}")
                                    st.write(f"ATS Score: {analysis_json.get('ats_score', 'Not Found')}")

                                    # Check if candidate matches the job role requirements
                                    if filter_candidate_based_on_skills_and_projects(analysis_json, job_role):
                                        st.success("This candidate is shortlisted!")
                                    else:
                                        st.warning("This candidate does not match the job role requirements.")

                                except json.JSONDecodeError:
                                    st.error("Error: Unable to parse resume analysis response.")
        else:
            st.error("Invalid Google Drive folder URL.")

if __name__ == "__main__":
    main()
