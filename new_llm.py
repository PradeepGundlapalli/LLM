import pandas as pd
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
import os
genai.configure(api_key=os.getenv("AIzaSyCgishs-1NQzIN2hHofqcv-sL1lM0k7LFI"))


# Load CSV into memory
csv_path = "telecom_expenses.csv"  # Update with actual path if needed
df = pd.read_csv(csv_path)

# Rename columns for consistency
df.columns = ["Company", "Year", "Total_Expense", "Call_Cost", "Data_Usage_GB"]
df["Year"] = df["Year"].astype(int)  # Ensure Year is an integer

# Initialize Google Gemini API
#genai.configure(api_key=os.getenv("AIzaSyCgishs-1NQzIN2hHofqcv-sL1lM0k7LFI"))  # Set API key in environment variable
#model = genai.GenerativeModel("gemini-1.5-pro")

# Initialize FastAPI app
app = FastAPI()

# Request body model
class QueryRequest(BaseModel):
    query: str

def get_expense(Company, year):
    """Fetch telecom expense from CSV based on Company and year."""
    result = df[(df["Company"].str.lower() == Company.lower()) & (df["Year"] == int(year))]

    if result.empty:
        print ("empty")
        return None  # No data found

    total_expense = result.iloc[0]["Total_Expense"]
    return f"The total telecom expense for {Company} in {year} is {total_expense}."

@app.post("/query")
def query_expense(request: QueryRequest):
    """Handles queries for telecom expenses."""
    query_text = request.query.lower()

    # Extract Company and year (basic keyword matching)
    words = query_text.split()
    print ("empty"+ str(words))
    Company = None
    year = None

    for word in words:
        if word.isdigit() and len(word) == 4:  # Identify year
            year = word
            print ("empty"+ str(year))
        elif word.capitalize() in df["Company"].unique():  # Identify Company
            Company = word.capitalize()

    if Company and year:
        expense_data = get_expense(Company, year)
        if expense_data:
            return {"response": expense_data}

    # If no data found, query Gemini AI
    #response = model.generate_content(request.query)
    #return {"response": response.text}

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
