"""
Defines Pydantic models for data validation and serialization.

These models are used to structure:
- Data returned by AI analysis functions (e.g., ResumeAnalysis).
- Results of similarity calculations (e.g., SimilarityResult).
- The final response body for API endpoints (e.g., AnalysisAndSimilarityResponse).

Pydantic models ensure data consistency and provide automatic validation.
"""

# Import necessary components from Pydantic and typing standard library
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Resume Analysis Schema ---

class ResumeAnalysis(BaseModel):
    """
    Defines the structured output expected from the AI resume analysis.
    Each field corresponds to a specific piece of information extracted or generated.
    """
    # Summary of the candidate's profile based on the resume text.
    summary: str = Field(description="A concise summary of the candidate's profile.")

    # List of key strengths identified in the resume.
    strengths: List[str] = Field(description="Bullet points highlighting key strengths of the candidate and their uploaded resume.")

    # List of potential weaknesses or areas lacking information in the resume.
    weaknesses: List[str] = Field(description="Bullet points highlighting potential weaknesses or areas missing info. in the candidate's resume. Also consider differences and gaps in requirements like location match if strictly required in the job description.")

    # List of key skills extracted directly from the resume text.
    skills: List[str] = Field(description="Key skills extracted from the resume.")

# --- Similarity Search Schemas ---

class RelevantChunk(BaseModel):
    """
    Represents a single chunk of text from the resume deemed relevant
    to the job description during similarity search, along with its metadata.
    """
    # The actual text content of the relevant resume chunk.
    content: str

    # The page number from the original PDF where the chunk originated (if available).
    # Optional[int] means this field can be an integer or None.
    page: Optional[int] = None

    # The similarity score calculated by the vector store (e.g., distance).
    # Interpretation depends on the vector store (lower is often better for distance metrics).
    score: float

class SimilarityResult(BaseModel):
    """
    Encapsulates the results of the resume vs. job description similarity analysis.
    """
    # An overall score representing similarity (e.g., average distance of top matches).
    # Interpretation depends on calculation method.
    overall_score: float

    # A potentially scaled similarity score represented as a percentage (0-100).
    # Note: The scaling method might be approximate and need tuning.
    similarity_percentage: float

    # A list containing the most relevant resume chunks found.
    relevant_chunks: List[RelevantChunk] # Uses the RelevantChunk model defined above.

# --- API Response Schema ---

class AnalysisAndSimilarityResponse(BaseModel):
    """
    Defines the structure of the final JSON response returned by the /analyze/ API endpoint.
    It combines the results of the resume analysis and the similarity comparison.
    """
    # Contains the structured analysis of the resume.
    # Nested model using the ResumeAnalysis schema.
    resume_analysis: ResumeAnalysis

    # Contains the results of the similarity calculation.
    # Nested model using the SimilarityResult schema.
    similarity_results: SimilarityResult

    # Optional field to report the total backend processing time in seconds.
    # Useful for performance monitoring or informing the user.
    processing_time_seconds: Optional[float] = None