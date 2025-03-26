import oracledb  # Oracle DB connector
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
import os
import re

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Set API key in environment variables

# Initialize FastAPI app
app = FastAPI()

# Oracle DB Connection Details
DB_USER = "PCLP25FEB25"
DB_PASSWORD = "PCLP25FEB25"
DB_DSN = "10.114.218.118:1521/emdev01"

# Function to get telecom expense data from Oracle DB
def get_expense(providername, year, month):
    """Fetch telecom expense from Oracle DB based on Company and year."""
    try:
        # Connect to Oracle DB
        with oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN) as conn:
            with conn.cursor() as cursor:
                # Query the Oracle view
                query = """
                    select  providername,
                            accountnumber,
                            accountname,
                           seriesname,
                           invoicenumber,
                           month,
                           year,
                           FORMITYSERVICENAME,
                           PROVIDERID,
                           custentityname,
                           controllocationname,
                           controllocationtype,
                           distantlocationname,
                           distantlocationtype,
                           chargebackstatus,
                           COST_CODE_FULL_PATH,
                           PARENTCOSTCENTER,
                           COSTCODE,
                           costentitytype,
                           charge,
                           contactname,
                           contactemployeeid,
                           contactstatus,
                           owned_by,
                           contactlocation,
                           contactlocationcity,
                           contactlocationstate,
                           contactlocationcountry,
                           providerservice,
                           company_name,
                           glcodename  
                          
                          from TELECOM_EXPENSES_VIEW

  
                    WHERE LOWER(providername) = :providername AND year = :year AND LOWER(month) = :month
                """
                print("1"+str(query))
                cursor.execute(query, {"providername": providername.lower(), "year": int(year), "month": month.lower()})
                result = cursor.fetchone()

                if result:
                    return f"The telecom expense for {result[0]} in month {result[5]} year {result[6]} is {result[19]}."
                else:
                    return f"No data found for the given query. "
    except Exception as e:
        return f"Database error: {str(e)}"

# Define request body model
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_expense(request: QueryRequest):
    """Handles queries for telecom expenses."""
    query_text = request.query.lower()

    # Fetch provider list from the database
    provider_list = get_providers_from_db()
    
    # Identify the provider from the provider list
    providername = next((p for p in provider_list if p.lower() in query_text), None)

  # Extract year and month
    year_match = re.search(r'\b(20\d{2})\b', query_text)  # Find year (e.g., 2023)
    month_match = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', query_text, re.IGNORECASE)

    year = year_match.group(1) if year_match else None
    month = month_match.group(1).capitalize() if month_match else None
 

    print("1"+str(providername))
    print("2"+str(month))
    print("3"+str(year))
    # Call function with extracted details
    expense_data = get_expense(providername, year, month)
    if expense_data:
     return {"response": expense_data}

    return {"response": "No data found for the given query."}

def get_providers_from_db():
    """Fetch provider names from the Oracle database."""
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN) as conn:
            with conn.cursor() as cursor:
                query = "SELECT provider_name FROM provider"
                cursor.execute(query)
                providers = [row[0] for row in cursor.fetchall()]  # Get all provider names
                return providers
    except Exception as e:
        print("Database error:", e)
        return []

        
# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
