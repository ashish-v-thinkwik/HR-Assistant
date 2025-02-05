import time
import streamlit as st
import os
import re
import pickle
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
import io
from openai import OpenAI
import tiktoken
from PyPDF2 import PdfReader
from docx import Document
import fitz  # PyMuPDF

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize tokenizer for GPT-4
encoding = tiktoken.encoding_for_model("gpt-4")

# Define the scopes your app will need
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Token management functions
def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def truncate_text(text: str, max_tokens: int) -> str:
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])

# Authenticate and retrieve documents from Google Drive
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
                'client_secret_979863872747-e4q396tetod1iaiv534hqdjogapjocu3.apps.googleusercontent.com.json', SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

# Retrieve job description documents from Drive
def get_documents_from_drive(drive_service, folder_id):
    query = f"'{folder_id}' in parents"
    results = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return results.get('files', [])

# Download file from Google Drive
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
    
    if isinstance(file_data, bytes):
        file_data = io.BytesIO(file_data)  # Convert raw bytes to BytesIO if necessary

    if file_extension == "pdf":
        try:
            doc = fitz.open(stream=file_data, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            truncated_text = truncate_text(text, 3000)
            return truncated_text
        except Exception as e:
            st.error(f"Error parsing PDF: {e}")
            return None

    elif file_extension == "docx":
        try:
            doc = Document(file_data)  # Directly use BytesIO object
            text = "\n".join(para.text for para in doc.paragraphs)
            truncated_text = truncate_text(text, 3000)
            return truncated_text
        except Exception as e:
            st.error(f"Error parsing DOCX: {e}")
            return None

    else:
        st.error("Unsupported file type.")
        return None

# Document summarization function
def summarize_document(text: str, max_tokens=500):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Summarize this document focusing on key requirements and skills. Keep under {max_tokens} tokens:\n{text}"
        }])
    return response.choices[0].message.content

# Optimized analysis function with token management
def analyze_with_context(resume_text, job_role, retrieved_documents, priority_skills, good_to_have_skills):
    MAX_TOKENS = 8192
    SYSTEM_MESSAGE_TOKENS = 100
    RESPONSE_TOKENS = 1024
    BUFFER_TOKENS = 200
    
    available_tokens = MAX_TOKENS - (SYSTEM_MESSAGE_TOKENS + RESPONSE_TOKENS + BUFFER_TOKENS)
    
    summarized_docs = [summarize_document(doc) for doc in retrieved_documents]
    context = ""
    context_tokens = 0
    
    for doc in summarized_docs:
        doc_tokens = count_tokens(doc)
        if context_tokens + doc_tokens <= available_tokens * 0.4:
            context += doc + "\n\n"
            context_tokens += doc_tokens
    
    resume_max_tokens = available_tokens - context_tokens
    processed_resume = truncate_text(resume_text, resume_max_tokens)
    
    prompt = (
        f"Job Role: {job_role}\n"
        f"Priority Skills: {', '.join(priority_skills)}\n"
        f"Good-to-Have Skills: {', '.join(good_to_have_skills)}\n"
        f"Context:\n{context}\n"
        f"Resume:\n{processed_resume}\n\n"
        "Analyze suitability. Provide JSON with:\n"
        "1. Candidate Name\n2. Relevant Skills\n3. Relevant Projects\n"
        "4. Experience Score (0-10)\n5. ATS Score\n"
        "Format: {name: string, skills: [], projects: [], project_skills:[], experience_score: int, ats_score: int}"
    )

    final_prompt_tokens = count_tokens(prompt)
    if final_prompt_tokens > available_tokens:
        prompt = truncate_text(prompt, available_tokens)

    messages = [
        {"role": "system", "content": "Expert resume analyst. Provide concise JSON responses."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=RESPONSE_TOKENS,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# Update the analysis response formatting
def process_resumes_in_batches(resumes, job_role, priority_skills, good_to_have_skills, batch_size=3):
    results = []
    total_resumes = len(resumes)
    st.write(f"Starting resume processing for {total_resumes} resumes.")
    
    # Track matches for prioritizing candidates
    priority_matches = []

    for i in range(0, len(resumes), batch_size):
        batch = resumes[i:i+batch_size]
        st.write(f"Processing batch {i//batch_size + 1} of {((total_resumes + batch_size - 1) // batch_size)}...")

        for resume in batch:
            resume_text = resume.get('text', ' ')
            if resume_text:
                analysis = analyze_with_context(resume_text, job_role, [], priority_skills, good_to_have_skills)
                if analysis:
                    try:
                        analysis_json = json.loads(analysis)
                        analysis_json["resume_name"] = resume["name"]
                        # Calculate matching priority skills
                        matched_priority_skills = [
                            skill for skill in priority_skills 
                            if skill in analysis_json.get('skills', []) or skill in analysis_json.get('project_skills', [])
                        ]
                        
                        analysis_json["matched_priority_skills"] = matched_priority_skills
                        analysis_json["priority_match_count"] = len(matched_priority_skills)
                        
                        # Append the result for later processing
                        results.append(analysis_json)
                        priority_matches.append({
                            'resume_name': resume['name'],
                            'priority_match_count': len(matched_priority_skills),
                            'matched_skills': matched_priority_skills
                        })
                    except json.JSONDecodeError:
                        st.error(f"Failed to parse analysis for {resume['name']}")
                
                time.sleep(1.5)  # Increased delay for safety

        time.sleep(6)  # Longer delay between batches

    # Shortlist candidates with the highest priority skill matches
    shortlisted = sorted(priority_matches, key=lambda x: x['priority_match_count'], reverse=True)

    # Display the shortlisted results based on priority matches

    return results

# Display results in structured format
def display_analysis_results(results):
    st.subheader("Analysis Results")
    
    # Shortlist candidates with priority matches
    shortlisted_results = [result for result in results if result.get("priority_match_count", 0) > 0]
    
    # Rejected candidates - no priority matches or very low match counts
    rejected_results = [result for result in results if result.get("priority_match_count", 0) == 0]

    # Display Shortlisted Candidates
    st.markdown("### 🟢 **Shortlisted Candidates**")
    for result in shortlisted_results:
        with st.expander(f"{result.get('name', 'Unnamed Resume')}"):
            # Basic information (Candidate Name, Shortlist/Reject, etc.)
            st.write(f"**Priority Match Count**: {result['priority_match_count']}")
            
            # Display "View Details" button for detailed information
            if st.button(f"View Details: {result.get('name', 'Unnamed Resume')}"):
                st.write("✅ **Priority Skills:**")
                for skill in result["skills"]:
                    st.write(f"- {skill}: ✅ Yes")
                
                st.write("🟡 **Good-to-Have Skills:**")
                for skill in result["skills"]:
                    st.write(f"- {skill}: ❌ No")

                st.write("🔷 **Project Skills Matching:**")
                for project in result["project_skills"]:
                    st.write(f"- {project}: ❌ No")  # Adjust logic if project skills are mapped separately
                
                st.write(f"### **Candidate Details:**")
                st.write(f"- **Name**: {result['name']}")
                st.write(f"- **Matched Skills**: {', '.join(result['skills'])}")
                st.write(f"- **Relevant Projects**: {', '.join(result['project_skills'])}")
                st.write(f"- **Experience Score**: **{result['experience_score']}/10**")
                st.write(f"- **ATS Score**: **{result['ats_score']}%**")
                st.write(f"- **Resume Name**: {result['resume_name']}")

    st.write("\n")

    # Display Rejected Candidates
    st.markdown("### ❌ **Rejected Candidates**")
    for result in rejected_results:
        with st.expander(f"{result.get('name', 'Unnamed Resume')}"):
            # Basic rejection information
            st.write(f"**Priority Match Count**: {result.get('priority_match_count', 0)}")
            st.write("No priority skills match found. Please review the resume further.")
            
            # Display "View Details" button for detailed information
            if st.button(f"View Details: {result.get('name', 'Unnamed Resume')} (Rejected)"):
                st.write(f"- **Name**: {result['name']}")
                st.write(f"- **ATS Score**: **{result.get('ats_score', 'N/A')}%**")
                st.write(f"- **Priority Match Count**: {result.get('priority_match_count', 0)}")
                st.write(f"- **Resume Name**: {result['resume_name']}")

    if not shortlisted_results:
        st.warning("No shortlisted candidates based on priority skill matches.")
    if not rejected_results:
        st.warning("No rejected candidates found.")

# Main function
def main():
    st.title("Resume Analysis Tool")

    # Input fields for URL, Job Role, Priority Skills, and Good-to-Have Skills
    folder_url = st.text_input("Enter Google Drive Folder URL:")
    job_role = st.text_input("Enter the Job Description Regarding the Job Requirement")
    
    priority_skills = st.text_input("Enter Priority Skills (comma separated):")
    good_to_have_skills = st.text_input("Enter Good to Have Skills (comma separated):")

    # Start processing only when all fields are filled
    if folder_url and job_role and priority_skills and good_to_have_skills:
        if st.button("Start Processing"):
            folder_id = re.search(r'\/folders\/([a-zA-Z0-9_-]+)', folder_url)
            if folder_id:
                drive_service = authenticate_google_drive()
                files = get_documents_from_drive(drive_service, folder_id.group(1))
                
                resumes = []
                for file in files:
                    if file['mimeType'] in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                        file_data = download_file(drive_service, file['id'], file['name'])
                        resume_text = parse_resume(file_data, file['name'])
                        if resume_text:
                            resumes.append({
                                'name': file['name'],
                                'text': resume_text
                            })

                if resumes:
                    with st.spinner("Analyzing resumes..."):
                        results = process_resumes_in_batches(resumes, job_role, priority_skills.split(','), good_to_have_skills.split(','))
                    
                    if results:
                        display_analysis_results(results)
                    else:
                        st.warning("No valid analysis results were generated.")
                else:
                    st.error("No valid resumes found in the folder.")
            else:
                st.error("Invalid Google Drive folder URL.")
        else:
            st.info("Fill in all fields and click 'Start Processing' to begin.")
    else:
        st.info("Please fill in all fields to start processing.")

if __name__ == "__main__":
    main()
