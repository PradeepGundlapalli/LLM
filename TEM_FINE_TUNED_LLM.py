import oracledb  # Oracle DB connector
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
import os
import re

 
# Initialize FastAPI app
app = FastAPI()

# Initialize gemini LLM

genai.configure(api_key= "AIzaSyAJqpA7IJWpUTJnIngtpewXea_A2RGJ1I0", transport="rest")
model = genai.GenerativeModel("gemini-1.5-pro")
    

# Oracle DB Connection Details
#DB_USER = "PCLP25FEB25"
DB_PASSWORD = "PCLP25FEB25"
DB_DSN = "10.114.218.118:1521/emdev01"

# Define request body model
class QueryRequest(BaseModel):
    query: str
    dbuser: str

# Define a function to generate SQL using LLM
def generate_sql(user_query,):
    """Use Gemini LLM to generate SQL query."""
        
    prompt = f"""
    Convert the following user request into an SQL query:

    Request: "{user_query}"

    Database Table: TELECOM_EXPENSES_VIEW
    Columns: company_name, accountname, accountnumber, invoicenumber, charge, providername, month, year, providerid, costcode, controllocationname

    - Return ONLY the SQL query, without any explanation.
    - Do NOT include markdown formatting (```sql ... ```)
    - Do NOT include semicolons at the end.
    
    
    Ensure:
    - Use `LOWER()` for providername , month, invoicenumber, accountnumber, costcode, controllocationname and providerid conditions.
    - Convert the year to an integer.
    - Return only relevant columns in the SELECT statement.

    Generate a valid SQL query without explanation.
    """
    response = model.generate_content(prompt)
    
    if response and response.text:
        sql_query = response.text.strip()
        
        # **Ensure NO markdown formatting and semicolons**
        sql_query = re.sub(r"```sql\n?", "", sql_query)  # Remove ```sql
        sql_query = re.sub(r"\n```", "", sql_query)  # Remove trailing ```
        sql_query = sql_query.strip().rstrip(";")  # Remove semicolons
        print(" after query " +str(sql_query));
        return sql_query
    return None

# Define a function to execute SQL on Oracle DB
def execute_sql(sql_query, dbuser):
    """Execute the generated SQL query in Oracle DB."""
    try:
        with oracledb.connect(user=dbuser, password=DB_PASSWORD, dsn=DB_DSN) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_query)
                result = cursor.fetchall()
                return result
    except Exception as e:
        return f"Database error: {str(e)}"

# API Endpoint to handle queries
@app.post("/query")
def query_expense(request: QueryRequest):
    """Handles natural language queries for telecom expenses."""
    user_query = request.query.lower()
    dbuser = request.dbuser.lower()
    print("dbuser  "+str(dbuser))
    # Generate SQL query from user input
    sql_query = generate_sql(user_query)

    if not sql_query:
        return {"response": "Failed to generate SQL query."}

    # Execute the SQL query
    result = execute_sql(sql_query,dbuser)
    formatted_response = generate_response(user_query, result)

    return {"query": sql_query, "response": formatted_response}

def generate_response(user_query, result):
    prompt = f"Convert the following database result into a natural language response: \n\nUser Query: {user_query}\nDatabase Result: {result}\n\nGenerate a response in a human-friendly format."
    
    response = model.generate_content(prompt)
    
    return response.text
    
# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
