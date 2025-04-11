from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import Dict, Any, Optional, List, Union
import json
import asyncio
import os
import uuid
from pydantic import BaseModel
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from azure.cosmos import CosmosClient, exceptions

# Import existing system
from loan_application_system import LoanApplication
from loan_application_api_adapter import LoanApplicationAdapter
from logging_config import configure_logging
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.DEBUG)

# Configure logging
logger = configure_logging()

# Create a global instance of LoanApplication
loan_application_instance = LoanApplication()

# Define question list
QUESTIONS = [
    {"field": "name", "question": "What is your name?"},
    {"field": "phone", "question": "Please provide your contact number."},
    {"field": "email", "question": "What is your email address?"},
    {"field": "address", "question": "What is your home address?"},
    {"field": "age", "question": "What is your age?"},
    {"field": "employer", "question": "Which company are you currently employed at?"},
    {"field": "position", "question": "What is your position?"},
    {"field": "monthly_income", "question": "What is your approximate monthly income?"},
    {"field": "loan_amount", "question": "How much loan amount do you wish to apply for?"},
    {"field": "loan_purpose", "question": "What is the purpose of the loan?"},
    {"field": "loan_term", "question": "What is the desired loan term in years?"},
    {"field": "property_address", "question": "What is the address of the mortgaged property?"},
    {"field": "property_size", "question": "What is the area of the property in square meters?"},
    {"field": "loan_start_date", "question": "When do you wish the loan to start?"},
    #{"field": "employment_certificate", "question": "Please upload your employment certificate using the button on the right."},
    #{"field": "bank_statement", "question": "Please upload your bank statement using the button on the right."}
]

# Define required fields
REQUIRED_FIELDS = {
    "name": "Name",
    "phone": "Phone",
    "email": "Email",
    "address": "Home Address",
    "age": "Age",
    "employer": "Employer",
    "position": "Position",
    "monthly_income": "Monthly Income",
    "loan_amount": "Loan Amount",
    "loan_purpose": "Loan Purpose",
    "loan_term": "Loan Term",
    "property_address": "Property Address",
    "property_size": "Property Size",
    "loan_start_date": "Loan Start Date"
}

# Define upload directory
UPLOAD_DIR = "uploads"

# Add Cosmos DB related configuration at the top of the file
COSMOS_ENDPOINT = os.getenv('COSMOS_ENDPOINT')
COSMOS_KEY = ""
# COSMOS_KEY = os.getenv('COSMOS_KEY')
COSMOS_DATABASE_NAME = os.getenv('COSMOS_DATABASE_NAME')
COSMOS_CONTAINER_NAME = os.getenv('COSMOS_CONTAINER_NAME')

# Create API application
app = FastAPI(title="Loan Evaluation System API", description="Solvay Capital Bank Loan Evaluation System API")

# Configure CORS - Use a more permissive configuration, ensure this is the first middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all response headers
    max_age=3600,         # Cache duration for preflight request results (seconds)
)

# Mount static files directory - Use the current directory as the static file root
app.mount("/static", StaticFiles(directory="."), name="static")

# Provide HTML chat interface
@app.get("/", response_class=FileResponse)
async def get_chat_interface():
    return FileResponse("loan_application_chatbot.html")

# Provide JavaScript client
@app.get("/loan_api_client.js", response_class=FileResponse)
async def get_api_client():
    return FileResponse("loan_api_client.js")

# Add root route
@app.get("/api", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Loan Evaluation System API</title>
        </head>
        <body>
            <h1>Welcome to the Solvay Capital Bank Loan Evaluation System API</h1>
            <p>API Documentation: <a href="/docs">/docs</a></p>
            <p>Health Check: <a href="/api/health">/api/health</a></p>
            <p>Chat Interface: <a href="/">/</a></p>
        </body>
    </html>
    """

# Session storage
sessions: Dict[str, Dict[str, Any]] = {}
files: Dict[str, str] = {}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    is_form_submit: bool = False
    form_data: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    session_id: str
    message: str
    state: str
    collected_data: Optional[Dict[str, Any]] = None
    conversation_state: Optional[Dict[str, Any]] = None

class LoanApplicationRequest(BaseModel):
    name: str
    age: int
    phone: str
    email: str
    address: str
    employer: str
    position: str
    monthly_income: float
    loan_amount: float
    loan_purpose: str
    loan_term: int
    property_address: str
    loan_start_date: str
    property_size: float
    employment_certificate: str
    bank_statement: str
    user_id: str

def initialize_session() -> str:
    """
    Initialize a new session
    
    Returns:
        str: New session ID
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "state": "welcome",
        "chat_history": [],
        "collected_data": {},
        "question_queue": [q for q in QUESTIONS],
        "current_question": None
    }
    return session_id

def get_next_question(session: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Get the next question
    
    Args:
        session: Session data
        
    Returns:
        Optional[Dict[str, str]]: The next question, returns None if no more questions
    """
    if not session["question_queue"]:
        return None
    session["current_question"] = session["question_queue"].pop(0)
    return session["current_question"]

def format_data_summary(data: Dict[str, Any]) -> str:
    """
    Format collected data into a readable string
    
    Args:
        data: Collected data dictionary
        
    Returns:
        str: Formatted data summary
    """
    summary = "Here is the information you provided:\n\n"
    for field, label in REQUIRED_FIELDS.items():
        value = data.get(field, "Not provided")
        summary += f"{label}: {value}\n"
    return summary

def validate_required_fields(data: Dict[str, Any]) -> List[str]:
    """
    Validate required fields
    
    Args:
        data: Collected data dictionary
        
    Returns:
        List[str]: List of missing fields
    """
    missing_fields = []
    for field in REQUIRED_FIELDS:
        if field not in data or not data[field]:
            missing_fields.append(REQUIRED_FIELDS[field])
    return missing_fields

# Add a generic exception handler to ensure all error responses include CORS headers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Server error: {str(exc)}", exc_info=True)
    status_code = 500
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
    
    return JSONResponse(
        status_code=status_code,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"}
    )
    
# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# OPTIONS preflight request handling
@app.options("/api/loan/chat")
async def options_chat():
    return {"status": "ok"}
    
# Define routes
@app.post("/api/loan/evaluate", response_model=Dict[str, Any])
async def evaluate_loan(application: LoanApplicationRequest):
    try:
        logger.info(f"Received loan application evaluation request: {application.name}")
        
        # Prepare data
        loan_data = {
            "Name": application.name,
            "Age": application.age,
            "Phone": application.phone,
            "Email": application.email,
            "Address": application.address,
            "Employer": application.employer,
            "Position": application.position,
            "Monthly Income": application.monthly_income,
            "Loan Amount": application.loan_amount,
            "Loan Purpose": application.loan_purpose,
            "Loan Term": application.loan_term,
            "Property Address": application.property_address,
            "Loan Start Date": application.loan_start_date,
            "Property Size": application.property_size,
            "employment_certificate": "ZhangSanZaiZhi.png",
            "bank_statement": "ZhangSanLiuShui.png",
            "user_id": application.user_id
        }
        # Asynchronously execute evaluation using the adapter
        loan_application_instance.collected_data = loan_data
        result = await LoanApplicationAdapter.evaluate_loan_async(loan_application_instance)
        logger.info(f"Loan evaluation completed: {application.name}, status: {result['status']}")
        
        # Save evaluation result to chat history
        session_id = str(uuid.uuid4())
        if "evaluation_details" in result:
            sessions[session_id] = {
                "evaluation_messages": result["evaluation_details"],
                "user_messages": []
            }
        
        return result
    except Exception as e:
        logger.error(f"Error during evaluation process: {str(e)}", exc_info=True)
        # Use the generic exception handler
        raise

# Chat endpoint - Use custom JSONResponse handler
@app.post("/api/loan/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        # Get or create session ID
        session_id = request.session_id or initialize_session()
        if session_id not in sessions:
            session_id = initialize_session()
        
        session = sessions[session_id]
        response_message = ""
        
        # Log user message
        session["chat_history"].append(ChatMessage(role="user", content=request.message))
        
        # Handle form submission
        if request.is_form_submit and request.form_data:
            session["collected_data"].update(request.form_data)
            missing_fields = validate_required_fields(session["collected_data"])
            
            if missing_fields:
                response_message = f"Please provide the following required information: {', '.join(missing_fields)}"
                session["state"] = "collecting"
            else:
                session["state"] = "evaluating"
                response_message = "Evaluating your loan application..."
        
        logger.info(f"session[state] : {session['state']}")

        # Handle based on session state
        if session["state"] == "welcome":
            response_message = "Welcome to the loan evaluation system! I will assist you in completing the loan application. Are you ready to start?"
            if "yes" in request.message or "okay" in request.message or "start" in request.message:
                session["state"] = "collecting"
                next_question = get_next_question(session)
                if next_question:
                    response_message = next_question["question"]
        
        elif session["state"] == "collecting":
            # Save user answer
            if session["current_question"]:
                session["collected_data"][session["current_question"]["field"]] = request.message
            
            # Get next question
            next_question = get_next_question(session)
            if next_question:
                response_message = next_question["question"]
            else:
                # All questions answered, display summary and request confirmation
                response_message = format_data_summary(session["collected_data"])
                response_message += "\n\nPlease upload the documents, if already uploaded, reply 'uploaded'."
                session["state"] = "confirming"
        
        elif session["state"] == "confirming":
            if request.message == "uploaded":
                session["state"] = "evaluating"
                response_message = "Solvay Capital Bank will use the above personal information for loan review, do you agree? (Agree/Disagree):"    
                # loan_application = LoanApplication(**session["collected_data"])
                # adapter = LoanApplicationAdapter()
                # evaluation_result = adapter.evaluate_loan_application(loan_application)
                # response_message = f"Evaluation result:\n{evaluation_result}"
                # session["state"] = "completed"
            # elif request.message == "disagree":
            #     response_message = "You have refused to use personal information for loan review, the application process is terminated. If you need to restart, please say 'restart'."
            #     session["state"] = "completed"
            else:
                response_message = "Please reply 'uploaded'."
        
        elif session["state"] == "evaluating":
            # Write to Cosmos DB
            # Create document
            document = setCosmosDB(session["collected_data"], session_id)
            # Write document to Cosmos DB
            await write_to_cosmos_db(document)
            
            # Evaluate loan application using backend system
            # loan_application = LoanApplication()#session["collected_data"]
            loan_application_instance.collected_data = session["collected_data"]
            adapter = LoanApplicationAdapter()
            # Use static method  
            # evaluation_result = await LoanApplicationAdapter.evaluate_loan_async(loan_application_instance)
            
            # response_message = f"Evaluation result:\n{evaluation_result}"
            # session["state"] = "completed"

            # TODO DB
        
        if session["state"] == "completed":
            response_message = response_message + "\n\n Your loan application has been processed. If you need to reapply, please say 'restart'."
            if "restart" in request.message or "start" in request.message:
                session_id = initialize_session()
                session = sessions[session_id]
                response_message = "Welcome to the loan evaluation system! I will assist you in completing the loan application. Are you ready to start?"
        
        # Log assistant response
        session["chat_history"].append(ChatMessage(role="assistant", content=response_message))
        
        return ChatResponse(
            session_id=session_id,
            message=response_message,
            state=session["state"],
            collected_data=session["collected_data"],
            conversation_state=session
        )
    
    except Exception as e:
        logger.error(f"Error occurred while processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# File upload endpoint
@app.post("/api/loan/upload/{user_id}")
async def upload_file(user_id: str, file: UploadFile = File(...)):
    try:
        # Create user directory
        user_dir = os.path.join(UPLOAD_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(user_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Update session state
        if user_id in sessions:
            sessions[user_id]["files_uploaded"] = True
        
        return {"message": "File uploaded successfully", "file_path": file_path}
    
    except Exception as e:
        logger.error(f"File upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Get session state
@app.get("/api/loan/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session does not exist")
    return sessions[session_id]

# Get chat history
@app.get("/api/loan/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Chat history does not exist")
    return sessions[session_id]["chat_history"]

async def async_initiate_chat(data_collector, user_proxy, message):
    # Use asyncio.to_thread to convert synchronous operation to asynchronous
    try:
        chat_result = await asyncio.to_thread(
            data_collector.initiate_chat,
            user_proxy,
            message=message,
            is_termination=lambda x: "DATA_COLLECTION_COMPLETE" in x.get("content", "")
        )
        return chat_result
    except Exception as e:
        logger.error(f"Asynchronous chat error: {str(e)}")
        return None
    
# File upload endpoint (optional)
@app.post("/api/loan/upload-documents")
async def upload_documents(
    user_id: str = Form(...),
    employment_certificate: UploadFile = File(None),
    bank_statement: UploadFile = File(None)
):
    try:
        logger.info(f"Received document upload request: User ID {user_id}")
        
        # Get Azure storage connection string
        connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not connect_str:
            raise ValueError("Azure Storage connection string not found")

        # Create BlobServiceClient instance
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        
        # Create container name (if it does not exist)
        container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME')
        try:
            container_client = blob_service_client.create_container(container_name)
        except ResourceExistsError:
            container_client = blob_service_client.get_container_client(container_name)

        result = {"uploaded_files": []}
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Upload employment certificate
        if employment_certificate:
            # Generate a unique blob name
            blob_name = f"employment_certificate_{current_time}_{employment_certificate.filename}"
            blob_client = container_client.get_blob_client(blob_name)
            
            # Read file content and upload
            content = await employment_certificate.read()
            blob_client.upload_blob(content, overwrite=True)
            
            result["uploaded_files"].append({
                "type": "employment_certificate",
                "filename": employment_certificate.filename,
                "blob_url": blob_client.url
            })
            files["employment_certificate"] = blob_name
            logger.info(f"Uploaded employment certificate to Azure Blob: {blob_name}")

        # Upload bank statement
        if bank_statement:
            # Generate a unique blob name
            blob_name = f"bank_statement_{current_time}_{bank_statement.filename}"
            blob_client = container_client.get_blob_client(blob_name)
            
            # Read file content and upload
            content = await bank_statement.read()
            blob_client.upload_blob(content, overwrite=True)
            
            result["uploaded_files"].append({
                "type": "bank_statement",
                "filename": bank_statement.filename,
                "blob_url": blob_client.url
            })
            files["bank_statement"] = blob_name
            logger.info(f"Uploaded bank statement to Azure Blob: {blob_name}")

        return result
    except Exception as e:
        logger.error(f"File upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

def setCosmosDB(data: Dict[str,Any],session_id:str) -> str:
 
    new_data = data.copy()
    # Add additional fields TODO
    new_data["id"] = str(session_id)  # Generate a new UUID as id
    new_data["applicantId"] = "123253433465"  # Example applicantId
    new_data["category"] = "loan-application"  # Example category
    new_data["jobTitle"] = data.get("position")  # Example category
    new_data["monthlyIncome"] = int(float(data.get("monthly_income"))) if data.get("monthly_income") is not None else None  # Convert to int
    new_data["loanAmount"] = int(float(data.get("loan_amount"))) if data.get("loan_amount") is not None else None  # Convert to int # Example category
    new_data["loanPurpose"] = data.get("loan_purpose")  # Example category
    new_data["loanTerm"] = int(data.get("loan_term")) if data.get("loan_term") is not None else None
    new_data["loanStartDate"] = data.get("loan_start_date") # Example category
    new_data["propertyArea"] = data.get("property_size")  # Example category
    new_data["propertyPrice"] = 88888  # Example category
    new_data["status"] = "evaluating"  # Example category
    new_data["submissionDate"] = datetime.utcnow().isoformat()  # Example category
    new_data["age"] = int(data.get("age")) if data.get("age") is not None else None  # Convert to int # Example category
    
    # Convert data dictionary to JSON string
    json_data = json.dumps(new_data, ensure_ascii=False)  # ensure_ascii=False to support Chinese characters
    
    logger.info(f"COSMOS DB: {json_data}")
    return json_data

async def write_to_cosmos_db(document: str):
    """
    Write document to Cosmos DB
    
    Args:
        document: JSON string to write
    """
    try:
        cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        database = cosmos_client.get_database_client(COSMOS_DATABASE_NAME)
        container = database.get_container_client(COSMOS_CONTAINER_NAME)

        # Write document to Cosmos DB
        container.upsert_item(json.loads(document))  # Convert JSON string to dictionary
        logger.info(f"Document information written to Cosmos DB: {document}")

    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"Failed to write to Cosmos DB: {str(e)}", exc_info=True)

# Start server
if __name__ == "__main__":
    # Ensure upload directory exists
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    logger.info("=== Solvay Capital Bank Loan Evaluation API Service Started ===")
    
    # Start service
    uvicorn.run("loan_api:app", host="0.0.0.0", port=8000, reload=True) 