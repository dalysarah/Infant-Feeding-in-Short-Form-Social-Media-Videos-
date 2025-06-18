# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:57:07 2024

@author: dalys
"""
#Install wheel and pandas in Anaconda terminal if program does not run 

#pip install wheel
#pip install pandas
#pip install google-generativeai
import pandas as pd
import google.generativeai as genai
import os

#https://github.com/google/generative-ai-docs/blob/main/site/en/gemini-api/docs/get-started/python.ipynb

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = ""  # TODO: Replace 'key' with your actual API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Path to the CSV file with the captions to analyze
csv_path = r'C:\DF_test.csv' #TODO: replace with the path to the CSV on your computer
df = pd.read_csv(csv_path)
#TODO replace with the path you would like to have assessed CSV saved to
output_csv = r'C:\analyzed_transcript_GENAI03.csv'
# Remove rows where the 'caption' column is empty or NaN
#df.dropna(subset=['caption'], inplace=True)
#df = df.dropna(subset=df.columns.values)
# display dataframe
print(df)
# Load the generative model once
model = genai.GenerativeModel("gemini-1.5-flash")


#Create analysis functions for your desired propmts
    
def analyze_text_theme(transcript):
    prompt = f"Is this video for a high risk baby?  {transcript}'"
    try:
        # Generate content using the model
        response = model.generate_content(prompt)  # Call generate on the model instance
        return response.text  # Access the result directly
    except Exception as e:
        print(f"Is this video for a high risk baby?: {transcript}\nError: {e}")
        return "Error"


# Apply the analysis functions to the 'caption' column
df['theme'] = df['Text'].apply(analyze_text_theme)

#yes/not output
#output2 = df.apply(analyze_text_theme)

# Save the DataFrame with responses to a new CSV file
#df.to_csv(output_csv, index=False)
#output2.to_csv(output_csv, index=False)