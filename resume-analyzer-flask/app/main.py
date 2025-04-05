"""
Defines the Flask web application for the AI Resume Analyzer API.

Handles incoming HTTP requests, routes them to appropriate logic,
interacts with backend services, and returns JSON responses.
"""

import time
from typing import Dict, Tuple, Any # Added Tuple and Any for type hints.

from flask import Flask, request, jsonify, abort # Core Flask components.
from flask_cors import CORS # Import CORS extension.
from werkzeug.utils import secure_filename # For handling filenames safely.

# Import schemas and services.
from .schemas import AnalysisAndSimilarityResponse # Response schema (used for typing and manual serialization).
from .services import (
    process_pdf_bytes,
    perform_resume_analysis,
    calculate_resume_jd_similarity,
    initialization_error # Check if components failed to load.
)

# --- Flask Application Instance ---
# Initialize the Flask application.
app = Flask(__name__)

# --- Configuration ---
# Optional: Load Flask specific config if needed, e.g., app.config['SECRET_KEY'].
# For this app, primary config (API Key) is handled via services/config.py.

# --- CORS (Cross-Origin Resource Sharing) ---
# Apply CORS to the Flask app to allow requests from the Streamlit frontend.
# Adjust origins for production. resource={r"/analyze/*":...} can target specific routes.
CORS(app, resources={r"/analyze/": {"origins": ["http://localhost:8501", "http://localhost"]}})
# Alternatively, allow all for local demo: CORS(app).

# --- Application Startup Check ---
# Flask doesn't have a direct equivalent to FastAPI's @app.on_event("startup")
# easily accessible here. We'll check 'initialization_error' within the request context.
# For more complex startup logic, Flask Blueprints or Application Factories could be explored.
if initialization_error:
     print(f"--- STARTUP WARNING (Flask App Level) ---")
     print(initialization_error)
     print("API endpoints might return 503 Service Unavailable.")
     print("------------------------------------------")


# --- API Endpoints ---

@app.route('/', methods=['GET'])
def read_root() -> Tuple[Dict[str, str], int]:
    """
    Root endpoint (GET /).

    Provides a simple status message indicating the API is running.
    Useful for basic health checks.
    """
    return jsonify({"status": "AI Resume Analyzer Flask API is running!"}), 200


@app.route('/analyze/', methods=['POST'])
def analyze_resume_and_jd_flask() -> Tuple[Dict[str, Any], int]:
    """
    Handles POST requests to analyze resume and job description PDFs.

    Expects 'resume_file' and 'jd_file' as multipart/form-data file uploads.
    Performs analysis and similarity comparison via backend services.

    Returns:
        JSON response conforming to AnalysisAndSimilarityResponse schema on success (200).
        JSON error response on failure (400, 422, 500, 503).
    """
    print("\nReceived request for /analyze/")
    start_time: float = time.time() # Record start time.

    # --- Initial Checks ---
    # Check if components failed to initialize during service module import.
    if initialization_error:
         print(f"ERROR: Initialization failed: {initialization_error}")
         # Use abort to raise standard HTTP exceptions in Flask.
         abort(503, description=f"API service unavailable due to initialization failure: {initialization_error}")

    # --- File Handling and Validation ---
    if 'resume_file' not in request.files or 'jd_file' not in request.files:
        print("ERROR: Missing 'resume_file' or 'jd_file' in request files.")
        abort(400, description="Missing 'resume_file' or 'jd_file' in the request.")

    resume_file = request.files['resume_file']
    jd_file = request.files['jd_file']

    # Basic filename validation.
    if not resume_file.filename or not jd_file.filename:
         print("ERROR: One or both uploaded files lack a filename.")
         abort(400, description="Uploaded files must have filenames.")

    # Secure filenames and check extensions.
    resume_filename = secure_filename(resume_file.filename)
    jd_filename = secure_filename(jd_file.filename)
    if not resume_filename.lower().endswith(".pdf") or not jd_filename.lower().endswith(".pdf"):
        print(f"ERROR: Invalid file type. Resume='{resume_filename}', JD='{jd_filename}'")
        abort(400, description="Invalid file type. Please upload PDF files only.")

    # --- Read File Bytes ---
    try:
        print(f"Reading file: {resume_filename}")
        resume_bytes: bytes = resume_file.read()
        print(f"Reading file: {jd_filename}")
        jd_bytes: bytes = jd_file.read()
        # Note: Flask's request.files handles closing implicitly in standard scenarios,
        # but it's generally good practice if you were manipulating the stream object directly.
    except Exception as e:
         print(f"ERROR reading files: {e}.")
         # Abort with 400 for bad requests (e.g., issues during read).
         abort(400, description=f"Failed to read file content: {e}.")

    print(f"Resume size: {len(resume_bytes)} bytes, JD size: {len(jd_bytes)} bytes")

    # --- Core Processing via Services ---
    try:
        # Call the same service functions as before.
        print("Processing Resume PDF...")
        resume_docs = process_pdf_bytes(resume_bytes, resume_filename)
        print("Processing Job Description PDF...")
        jd_docs = process_pdf_bytes(jd_bytes, jd_filename)

        print("Performing resume analysis...")
        analysis_result = perform_resume_analysis(resume_docs)

        print("Calculating similarity...")
        similarity_result = calculate_resume_jd_similarity(resume_docs, jd_docs)

    except ValueError as ve:
         # Handle specific errors raised by services (e.g., processing failed).
         print(f"ERROR during processing: {ve}")
         # Return 422 Unprocessable Entity for validation/logical errors from services.
         # Manually create JSON error response.
         return jsonify({"error": "Processing Error", "detail": str(ve)}), 422
    except Exception as e:
         # Handle unexpected errors during service calls
         print(f"UNEXPECTED INTERNAL ERROR during processing: {e}.")
         # TODO: Log traceback for debugging.
         # Return 500 Internal Server Error.
         return jsonify({"error": "Internal Server Error", "detail": str(e)}), 500

    # --- Format and Return Success Response ---
    end_time: float = time.time()
    processing_time: float = end_time - start_time
    print(f"Request processing finished successfully in {processing_time:.2f} seconds.")

    # Manually construct the response structure (can use Pydantic model for structure).
    response_data = AnalysisAndSimilarityResponse(
        resume_analysis=analysis_result,
        similarity_results=similarity_result,
        processing_time_seconds=processing_time
    )

    # Convert Pydantic model to dict and then jsonify.
    # Use model_dump() for Pydantic v2+, use dict() for v1.
    try:
        response_dict = response_data.model_dump(mode='json') # Pydantic V2+.
    except AttributeError:
        response_dict = response_data.dict() # Pydantic V1 fallback.

    return jsonify(response_dict), 200


# --- Run Flask Development Server ---
# This block allows running the Flask app directly using `python app/main.py`.
# Not recommended for production, use WSGI servers like Gunicorn or Waitress instead.
if __name__ == '__main__':
    # host='0.0.0.0' makes it accessible on your network.
    # debug=True enables auto-reloading and provides detailed error pages (DO NOT use in production).
    # port=8000 matches the port expected by the Streamlit app.
    app.run(host='0.0.0.0', port=8000, debug=True)