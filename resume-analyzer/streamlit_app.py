"""
Streamlit frontend application for the AI Resume Analyzer.

This application provides a user interface for uploading a resume and job description (PDFs).
It communicates with a separate FastAPI backend to perform AI-driven analysis and
similarity comparison, then displays the results to the user.
"""

import streamlit as st
import requests # Library to make HTTP requests to the FastAPI backend.
from typing import Optional, Dict, Any, Tuple, List # For type hinting.
from streamlit.runtime.uploaded_file_manager import UploadedFile # Type hint for uploaded files.
from requests.models import Response # Type hint for requests response object.

# --- Configuration ---
# The URL where the FastAPI backend's '/analyze/' endpoint is running.
# Ensure this matches the host and port where you run uvicorn.
FASTAPI_URL: str = "http://127.0.0.1:8000/analyze/"

# --- Streamlit Page Setup ---
# Configure the page title, icon, and layout.
st.set_page_config(page_title="AI Resume Analyzer", layout="wide", initial_sidebar_state="collapsed")

# --- Application UI ---
st.title("📄 AI Resume Analyzer Demo")
st.markdown("""
Upload a candidate's resume and a job description (both in PDF format).
The application will call a backend API to analyze the resume and compare it against the job description using AI.
""")
# Add a note indicating the backend dependency
st.caption("*(Ensure the FastAPI backend server is running)*")

# Use columns for a side-by-side layout for file uploaders.
col1, col2 = st.columns(2)

# Declare variables for uploaded files with type hints.
# Using Optional because they are None until the user uploads a file.
resume_file: Optional[UploadedFile] = None
jd_file: Optional[UploadedFile] = None

with col1:
    # Streamlit file uploader widget for the resume.
    # - label: Text displayed above the uploader.
    # - type: Specifies allowed file extensions.
    # - key: A unique identifier for this widget state.
    resume_file = st.file_uploader("1. Upload Resume PDF", type="pdf", key="resume")
with col2:
    # Streamlit file uploader widget for the job description.
    jd_file = st.file_uploader("2. Upload Job Description PDF", type="pdf", key="jd")

# Streamlit button widget to trigger the analysis.
# - label: Text displayed on the button.
# - type: "primary" makes the button visually distinct.
# - disabled: Control whether the button is clickable. It's disabled if either file is not uploaded.
analyze_button: bool = st.button(
    "Analyze and Compare",
    type="primary",
    disabled=(not resume_file or not jd_file) # Button is disabled until both files are uploaded.
)

# --- Processing Logic ---
# This block executes only when the 'Analyze and Compare' button is clicked.
if analyze_button:
    # Double-check if files are present (though the button 'disabled' state should prevent this).
    if resume_file is not None and jd_file is not None:
        # Show a spinner animation while processing occurs.
        with st.spinner("🚀 Sending files to backend and analyzing... This might take a minute."):
            try:
                # --- Prepare Files for API Request ---
                # Create a dictionary for the 'files' parameter of requests.post.
                # The format is {'form_field_name': (filename, file_bytes, content_type)}.
                # 'form_field_name' must match the parameter names in the FastAPI endpoint function.
                files: Dict[str, Tuple[Optional[str], bytes, Optional[str]]] = {
                    "resume_file": (resume_file.name, resume_file.getvalue(), resume_file.type),
                    "jd_file": (jd_file.name, jd_file.getvalue(), jd_file.type),
                }

                # --- Make API Call ---
                st.info(f"📞 Contacting API backend at {FASTAPI_URL}...")
                # Send a POST request to the FastAPI endpoint with the files.
                # - timeout: Set a timeout (in seconds) for the request to prevent indefinite waiting. Adjust as needed.
                response: Response = requests.post(FASTAPI_URL, files=files, timeout=300)

                # --- Handle API Response ---
                # Check if the API request was successful (HTTP status code 200 OK).
                if response.status_code == 200:
                    st.success("✅ Analysis Complete!")
                    # Parse the JSON data returned by the FastAPI backend.
                    results: Dict[str, Any] = response.json()

                    # --- Display Resume Analysis ---
                    st.subheader("📊 Resume Analysis Results")
                    # Safely access nested dictionary data using .get() with default values.
                    analysis: Dict[str, Any] = results.get("resume_analysis", {})
                    st.markdown(f"**Summary:**\n {analysis.get('summary', 'Not Available')}")

                    # Use expanders to neatly display list-based results.
                    with st.expander("✅ Strengths"):
                        strengths: List[str] = analysis.get('strengths', [])
                        if strengths:
                            for item in strengths: st.markdown(f"- {item}")
                        else:
                            st.markdown("No specific strengths identified by the AI.")

                    with st.expander("⚠️ Weaknesses / Areas for Improvement"):
                        weaknesses: List[str] = analysis.get('weaknesses', [])
                        if weaknesses:
                            for item in weaknesses: st.markdown(f"- {item}")
                        else:
                            st.markdown("No specific weaknesses or missing areas identified by the AI.")

                    with st.expander("🛠️ Identified Skills"):
                         skills: List[str] = analysis.get('skills', [])
                         if skills:
                             # Display skills as a comma-separated string or bullet points.
                             st.markdown(", ".join(skills))
                         else:
                             st.markdown("No specific skills identified by the AI.")

                    # --- Display Similarity Results ---
                    st.subheader("📈 Resume vs. Job Description Similarity")
                    similarity: Dict[str, Any] = results.get("similarity_results", {})
                    # Safely get similarity metrics.
                    overall_score: float = similarity.get('overall_score', float('inf')) # Default to infinity if not found.
                    sim_perc: float = similarity.get('similarity_percentage', 0.0) # Default to 0% if not found.

                    # Display the raw score (lower is better) and the approximate percentage.
                    st.metric(label="Similarity Score (Lower is Better - FAISS L2 Distance)", value=f"{overall_score:.4f}")
                    # Display the approximate percentage match using a progress bar.
                    st.progress(sim_perc / 100.0, text=f"Approx. Match: {sim_perc:.2f}% (Note: Scaling is approximate!)")

                    st.markdown("**Most Relevant Resume Sections (based on JD similarity):**")
                    chunks: List[Dict[str, Any]] = similarity.get('relevant_chunks', [])
                    if chunks:
                        # Display each relevant chunk in an expander.
                        for i, chunk_data in enumerate(chunks):
                            chunk_score = chunk_data.get('score', 0.0)
                            chunk_page = chunk_data.get('page', 'N/A')
                            chunk_content = chunk_data.get('content', 'N/A')
                            with st.expander(f"Match {i+1} (Score: {chunk_score:.4f} | Page: {chunk_page})"):
                                # Use st.caption for smaller text, suitable for content snippets.
                                st.caption(chunk_content)
                    else:
                        st.markdown("No relevant sections identified based on similarity search.")

                    # --- Display Optional Processing Time ---
                    proc_time: Optional[float] = results.get("processing_time_seconds")
                    if proc_time is not None:
                        st.caption(f"Backend processing time: {proc_time:.2f} seconds.")

                # --- Handle Non-200 HTTP Status Codes ---
                else:
                    st.error(f"❌ API Error: Received Status Code {response.status_code}")
                    try:
                        # Attempt to parse error details from the JSON response body.
                        error_detail: str = response.json().get("detail", response.text)
                        st.error(f"Details: {error_detail}")
                    except requests.exceptions.JSONDecodeError:
                        # Fallback if the response body is not valid JSON.
                        st.error(f"Could not decode error response body: {response.text}")

            # --- Handle Network/Connection Errors ---
            except requests.exceptions.RequestException as e:
                st.error(f"🔗 Network Error: Could not connect to the API backend at {FASTAPI_URL}.")
                st.error(f"Details: {e}")
                st.info("Troubleshooting Tips: \n1. Is the FastAPI server running? \n2. Is the URL correct? \n3. Check for network issues.")
            # --- Handle Other Unexpected Errors ---
            except Exception as e:
                st.error("🔥 An unexpected error occurred in the Streamlit application.")
                # Display the full exception traceback for debugging.
                st.exception(e)
    else:
        # This message is shown if the button is clicked but somehow one of the files is None.
        # Should be less common due to the button's disabled state, but good as a fallback.
        st.warning(" P lease upload both Resume and Job Description PDF files before clicking Analyze.")