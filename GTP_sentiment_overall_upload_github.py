# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:48:17 2025

@author: dalys
"""

#    pip install openai
#`pip install openai==0.28`
import openai
import pandas as pd
import time
  
openai.api_key = ""

#Path to the CSV file with the captions to analyze
csv_path = r'C:\text_sentiment.csv' #TODO: replace with the path to the CSV on your computer
df = pd.read_csv(csv_path)
#TODO replace with the path you would like to have assessed CSV saved to
output_csv = r'C:\analyzed_transcript_sent_GTP_2.csv'
# Remove rows where the 'caption' column is empty or NaN
#df.dropna(subset=['caption'], inplace=True)
#df = df.dropna(subset=df.columns.values)
# display dataframe
print(df)
#print(df['question'][1])
print(df['Text'])
# Load the generative model once
#model = openai.chat.completions.create("gpt-3.5-turbo")
print('okay!')

# Function to analyze text using OpenAI API with retries
#Add five second dealy 
def analyze_text_theme(transcript,prompt, max_retries=1, initial_delay=1):
    
    prompt2 = f"{prompt} {transcript}"

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt2}]
            )
            return response["choices"][0]["message"]["content"]  # Corrected response access

        except openai.OpenAIError as e:  # Fixed AttributeError
            if "insufficient_quota" in str(e):
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                print(f"Quota exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"OpenAI API Error: {e}")
                return "API Error"

        except Exception as e:
            print(f"Unexpected error: {e}")
            return "Error"

    return "Max retries exceeded. Unable to complete the request."

print('okay2!')
# Apply the function to the 'Text' column
#df['theme'] = df['Text'].apply(analyze_text_theme)
# Save the DataFrame with responses to a new CSV file
#df.to_csv(output_csv, index=False)

# Process test dataset
data2 = []
print('okay3!')

#result = analyze_text_theme(df['Text'][0:454], "overall, what are the prevalent tones of these videos?")
#result = analyze_text_theme(df['Text'][0:454], "do these videos present formula or breastfeeding as overwhelming or stressful?")
#result = analyze_text_theme(df['Text'][0:454], "overall, what are the prevalent purposes of these videos?")
#result = analyze_text_theme(df['Text'][0:454], "overall, what are the main challenges these videos discuss?")
#result = analyze_text_theme(df['Text'][0:454], "overall, are these videos educational or commecial?")
#result = analyze_text_theme(df['Text'][0:454], "overall, what percentage of videos are commecial?")
#result = analyze_text_theme(df['Text'][0:454], "Do do the creators present the sanitation of infant feeding equipment as time consuming?")
#result = analyze_text_theme(df['Text'][0:454], "Do do the creators present the cleaning of infant feeding equipment as overwhelming or stressful?")
#result = analyze_text_theme(df['Text'][0:454], "overall, does the text indicate poor mental health?")
#result = analyze_text_theme(df['Text'][0:454], "overall, what are the main stressors presented in this text?")
#result = analyze_text_theme(df['Text'][0:454], "overall, is the text discussing community building?")
#result = analyze_text_theme(df['Text'][0:454], "overall, are creators giving advice?")
#result = analyze_text_theme(df['Text'][0:454], "overall, are creators seeking advice?")
result = analyze_text_theme(df['Text'][0:454], "overall, do the texts imply that breastfeeding is better than formula feeding?")

# Append the result to the list as a dictionary
data2.append({'result': result})

data2 = pd.DataFrame(data2)
data2.to_csv('GPT_sentiment_output_overall.csv', index=False)

print(data2)
