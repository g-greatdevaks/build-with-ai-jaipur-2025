"""
Defines the FastAPI application, endpoints, middleware, and startup logic
for the AI Resume Analyzer API.
"""

import time
from typing import Dict, List # Added Dict and List for type hinting.

# FastAPI core components and utilities.
from fastapi import (
    FastAPI,
    File,        # Used for declaring file upload parameters.
    UploadFile,  # The type hint for uploaded files.
    HTTPException, # Used for returning standard HTTP errors.
    status       # Contains standard HTTP status codes.
)
# Middleware for handling Cross-Origin Resource Sharing (CORS).
from fastapi.middleware.cors import CORSMiddleware

# Import schemas for request/response validation and serialization.
from .schemas import AnalysisAndSimilarityResponse, ResumeAnalysis, SimilarityResult # Added nested schemas for hints.

# Import core processing functions from the services module.
from .services import (
    process_pdf_bytes,
    perform_resume_analysis,
    calculate_resume_jd_similarity,
    initialization_error # Check if components failed to load during startup.
)

# Import Document type for hinting.
from langchain_core.documents import Document

# --- FastAPI Application Instance ---
# Initialize the FastAPI application with metadata.
app = FastAPI(
    title="AI Resume Analyzer API",
    description="API for analyzing resumes against job descriptions using LangChain and Gemini.",
    version="0.1.0",
    # You can add more metadata like terms_of_service, contact, license_info, etc.
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# This allows the Streamlit frontend (running on a different port/origin)
# to make requests to this FastAPI backend.
origins = [
    "http://localhost",
    "http://localhost:8501", # Default Streamlit development port.
    # Add any other origins if needed (e.g., deployed frontend URL).
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins allowed to make requests.
    # Set allow_origins=["*"] for simplicity during demo/local dev if needed,
    # BUT BEWARE: this is insecure for production environments.
    allow_credentials=True, # Allow cookies if needed (not used here).
    allow_methods=["*"],    # Allow all standard HTTP methods (GET, POST, etc.).
    allow_headers=["*"],    # Allow all request headers.
)

# --- Application Event Handlers ---

@app.on_event("startup")
async def startup_event() -> None:
    """
    Asynchronous function executed once when the FastAPI application starts.
    Used here to check and log the initialization status of critical components.
    """
    print("FastAPI application startup commencing...")
    if initialization_error:
        # Log a warning if components (LLM, Embeddings, etc.) failed to initialize.
        print(f"--- STARTUP WARNING ---")
        print(initialization_error)
        print("-----------------------")
    else:
        # Confirm successful initialization.
        print("Google AI components appear to be initialized successfully.")
    print("Startup complete. API is ready.")


# --- API Endpoints ---

@app.get("/", tags=["General"])
async def read_root() -> Dict[str, str]:
    """
    Root endpoint (GET /).

    Provides a simple status message indicating the API is running.
    Useful for basic health checks. Added to the 'General' tag group in API docs.
    """
    return {"status": "AI Resume Analyzer API is running!"}


@app.post(
    "/analyze/",
    response_model=AnalysisAndSimilarityResponse, # Defines the structure of a successful response.
    tags=["Analysis"], # Groups this endpoint under 'Analysis' in API docs.
    summary="Analyze Resume and Job Description PDFs." # Short summary for API docs.
)
async def analyze_resume_and_jd(
    resume_file: UploadFile = File(..., description="The candidate's resume PDF file."),
    jd_file: UploadFile = File(..., description="The job description PDF file.")
) -> AnalysisAndSimilarityResponse:
    """
    Handles the upload of resume and job description PDF files, orchestrates
    the analysis and similarity comparison using backend services, and returns
    the combined results.

    Args:
        resume_file: The uploaded resume PDF file. Sent as part of a multipart/form-data request.
        jd_file: The uploaded job description PDF file. Sent as part of a multipart/form-data request.

    Returns:
        An AnalysisAndSimilarityResponse object containing the structured results
        of the resume analysis and similarity comparison.

    Raises:
        HTTPException(400): If files are not PDFs or cannot be read.
        HTTPException(422): If processing fails due to invalid data or service errors (ValueError from services).
        HTTPException(500): For unexpected internal server errors during processing.
        HTTPException(503): If critical AI components failed to initialize on startup.
    """
    print("\nReceived request for /analyze/.")
    start_time: float = time.time() # Record start time for performance monitoring.

    # --- Initial Checks ---
    # Immediately check if critical components failed during startup.
    if initialization_error:
         print(f"ERROR: Attempted access but initialization failed: {initialization_error}.")
         # Return 503 Service Unavailable if backend components are down.
         raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"API service unavailable due to initialization failure: {initialization_error}."
         )

    # --- File Validation and Reading ---
    # Basic validation: Check file extensions.
    if not resume_file.filename or not resume_file.filename.lower().endswith(".pdf") or \
       not jd_file.filename or not jd_file.filename.lower().endswith(".pdf"):
        print(f"ERROR: Invalid file type. Resume='{resume_file.filename}', JD='{jd_file.filename}'")
        # Return 400 Bad Request for invalid file types.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload PDF files only for both resume and job description."
        )

    resume_bytes: bytes
    jd_bytes: bytes
    try:
        # Read file contents asynchronously into memory as bytes.
        print(f"Reading resume file: {resume_file.filename}.")
        resume_bytes = await resume_file.read()
        print(f"Reading job description file: {jd_file.filename}.")
        jd_bytes = await jd_file.read()
    except Exception as e:
         # Handle potential errors during file reading.
         print(f"ERROR reading files: {e}.")
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file content: {e}."
         )
    finally:
        # --- Important: Close uploaded files ---
        # It's good practice to explicitly close files after reading.
        await resume_file.close()
        await jd_file.close()

    print(f"Resume size: {len(resume_bytes)} bytes, JD size: {len(jd_bytes)} bytes.")

    # --- Core Processing via Services ---
    try:
        # Step 1: Process PDFs into LangChain Document chunks.
        print("Processing Resume PDF...")
        resume_docs: List[Document] = process_pdf_bytes(resume_bytes, resume_file.filename)
        print("Processing Job Description PDF...")
        jd_docs: List[Document] = process_pdf_bytes(jd_bytes, jd_file.filename)

        # Note: Service functions now handle cases where docs might be empty.

        # Step 2: Perform AI analysis on the resume.
        print("Performing resume analysis...")
        analysis_result: ResumeAnalysis = perform_resume_analysis(resume_docs)

        # Step 3: Calculate similarity between resume and JD.
        print("Calculating similarity...")
        similarity_result: SimilarityResult = calculate_resume_jd_similarity(resume_docs, jd_docs)

    except ValueError as ve:
         # Catch specific errors raised by service functions (e.g., processing failures).
         print(f"ERROR during processing: {ve}")
         # Return 422 Unprocessable Entity for logical/data errors during processing.
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
         )
    except Exception as e:
         # Catch any other unexpected errors during the service calls.
         print(f"UNEXPECTED INTERNAL ERROR during processing: {e}")
         # TODO: Log the full traceback here for detailed debugging in production.
         # import traceback; traceback.print_exc(); # Example for logging.
         # Return 500 Internal Server Error for unexpected issues.
         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during processing: {e}"
         )

    # --- Finalize and Return Response ---
    end_time: float = time.time()
    processing_time: float = end_time - start_time
    print(f"Request processing finished successfully in {processing_time:.2f} seconds.")

    # Construct the final response object using the Pydantic model.
    # FastAPI automatically serializes this Pydantic object to JSON.
    return AnalysisAndSimilarityResponse(
        resume_analysis=analysis_result,
        similarity_results=similarity_result,
        processing_time_seconds=processing_time
    )