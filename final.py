import streamlit as st
import PyPDF2
from deep_translator import GoogleTranslator
import time
import google.generativeai as genai
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from io import BytesIO
from dotenv import load_dotenv
import threading
import openai
from typing import List

# Load environment variables
load_dotenv()

# Get API key from environment variable

api_key = os.getenv('OPENAI_API_KEY')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    st.write(f"PDF has {len(reader.pages)} pages")
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += page_text
        st.write(f"Extracted {len(page_text)} characters from page {i+1}")
    st.write(f"Total extracted text: {len(text)} characters")
    return text

# Function to translate large text
def translate_large_text(text, chunk_size=4500):
    translator = GoogleTranslator(source='es', target='en')
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    translated_text = ""
    
    st.write(f"Splitting text into {len(chunks)} chunks for translation")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        translated_chunk = translator.translate(chunk)
        translated_text += translated_chunk
        time.sleep(1)  # To avoid hitting rate limits
        
        # Update progress
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
        status_text.text(f"Translating: {int(progress * 100)}% complete. Chunk {i+1}/{len(chunks)}")
        
        st.write(f"Chunk {i+1}: Translated {len(chunk)} characters to {len(translated_chunk)} characters")
    
    status_text.text("Translation completed!")
    st.write(f"Total translated text: {len(translated_text)} characters")
    return translated_text

def split_text(text: str, max_tokens: int = 4000) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Function to extract product information and summary 
def extract_info_and_summarize(text):
    st.write("Preparing text for GPT-4 analysis and summarization")
    st.write(f"Input text length: {len(text)} characters")

    chunks = split_text(text)
    st.write(f"Split text into {len(chunks)} chunks")

    summaries = []
    for i, chunk in enumerate(chunks):
        st.write(f"Processing chunk {i+1}/{len(chunks)}")
        
        prompt = f"""
        Analyze the following part of a pharmaceutical product information and provide a partial structured summary:

        # Partial Analysis of Pharmaceutical Product

        [Provide relevant information for any of the following categories that are present in this text chunk. 
        If a category is not addressed in this chunk, omit it from the summary.]

        1. Product Name
        2. Brief Description
        3. Composition (Active ingredients)
        4. Excipients with Known Effects
        5. Dosage Form
        6. Posology
        7. Warnings
        8. Other Essential Information

        Text chunk: {chunk}

        Note: Provide the output with headings should be in bold and normal text size for the content.
        """
        from openai import OpenAI
        client =  OpenAI(api_key=api_key)
        response =  client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes pharmaceutical product information and provides structured summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )

        summary = response.choices[0].message['content']
        summaries.append(summary)
        st.write(f"Processed chunk {i+1}. Response length: {len(summary)} characters")

    # Combine summaries
    combined_summary = "\n\n".join(summaries)

    # Generate final summary
    final_prompt = f"""
    Based on the following combined partial analyses of a pharmaceutical product, provide a final structured summary:

    {combined_summary}

    Please format the final summary as follows:

    Analysis of Pharmaceutical Product(normal text size but in bold)

    1. Product Name
    [Product Name which is just 2 words]

    2. Brief Description
    [2-3 sentence description of the product]

    3. Composition
    - Active ingredient(s):
      - [Ingredient 1]
      - [Ingredient 2]
      - ...

    4. Excipients with Known Effects
    - [Excipient 1]
    - [Excipient 2]
    - ...

    5. Dosage Form
    [Dosage form]

    6. Posology
    [Brief summary of dosage instructions]

    7. Warnings
    - [Warning 1]
    - [Warning 2]
    - ...

    8. Other Essential Information
    - [Information 1]
    - [Information 2]
    - ...

    9. Overall Summary
    [A concise summary of the key points from the text]

    Note: Provide the output with headings should be must in bold and normal text size for the content.
    """
    from openai import OpenAI
    client =  OpenAI(api_key=api_key)
    final_response =  client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes pharmaceutical product information and provides structured summaries."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.5,
        max_tokens=1000
    )

    final_summary = final_response.choices[0].message['content']
    st.write(f"Generated final summary. Length: {len(final_summary)} characters")
    return final_summary

def generate_difference_table(info1, info2):
    prompt = f"""
    Create a difference table for the following two pharmaceutical products. Use the product names as the top heading. For each key information heading, provide a concise comparison of the two products.

    Product 1 Information:
    {info1}

    Product 2 Information:
    {info2}
    

    Format the output as a markdown table.
    
    After the table, provide the following sections as regular text (not in table format):

    Advantages and Competitive Edge:
     - List the advantages and competitive edge of Azacitidine Sandoz
     - List the advantages and competitive edge of Azacitidine Ever Pharma

    Overall conclusion
     - Provide an overall evaluation and recommendation on which product is better based on the analysis above.
     - Summarize the findings, emphasizing the distinct advantages and competitive edges of each product, and provide a final recommendation.
     
    Give headings in bold letters 
    """
    
    from openai import OpenAI
    client =  OpenAI(api_key=api_key)
    response =  client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates concise difference tables for pharmaceutical products."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

def create_pdf(content):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    for line in content.split('\n'):
        p = Paragraph(line, styles['Normal'])
        flowables.append(p)

    doc.build(flowables)
    buffer.seek(0)
    return buffer

def process_pdf(file, result_dict, key):
    try:
        st.write(f"Starting to process {key}")
        st.write(f"File size: {file.size} bytes")
        
        spanish_text = extract_text_from_pdf(file)
        st.write(f"{key}: Extracted text. Length: {len(spanish_text)} characters")
        
        english_text = translate_large_text(spanish_text)
        st.write(f"{key}: Translated text. Length: {len(english_text)} characters")
        
        # Display a sample of the translated text
        st.write(f"Sample of translated text for {key}:")
        st.write(english_text[:500] + "...")  # Display first 500 characters
        
        info_and_summary = extract_info_and_summarize(english_text)
        st.write(f"{key}: Generated summary. Length: {len(info_and_summary)} characters")
        
        result_dict[key] = {
            'english_text': english_text,
            'info_and_summary': info_and_summary
        }
        st.write(f"Finished processing {key}")
    except Exception as e:
        st.error(f"Error processing {key}: {str(e)}")
        result_dict[key] = {
            'english_text': f"Error: {str(e)}",
            'info_and_summary': f"Error: {str(e)}"
        }

def extract_product_name_openai(info):
    prompt = f"""
    Extract the product name from the following information. Return only the product name, nothing else.

    Information:
    {info}
    """
    from openai import OpenAI
    client =  OpenAI(api_key=api_key)
    response =  client.chat.completions.create(
        model="gpt-4",  # You can use "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts product name which is just 2 to 3 words from pharmaceutical information."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content'].strip()

# Streamlit app
st.title("SMPC summarization in English")

# Multiple file uploader
uploaded_files = st.file_uploader("Choose two PDF files", type="pdf", accept_multiple_files=True)

st.markdown("""
<style>
    .product-analysis h1 {font-size: 24px;}
    .product-analysis h2 {font-size: 20px;}
    .product-analysis p {font-size: 16px;}
    .product-analysis ul {font-size: 16px;}
    .product-analysis li {font-size: 16px;}
    .product-analysis strong {font-weight: bold;}
    .product-analysis em {font-style: italic;}
</style>
""", unsafe_allow_html=True)

if 'results' not in st.session_state:
    st.session_state.results = {}

if uploaded_files and len(uploaded_files) == 2 and api_key:
    if st.button("Process PDFs"):
        try:
            with st.spinner("Processing PDFs..."):
                results = {}
                threads = [
                    threading.Thread(target=process_pdf, args=(uploaded_files[0], results, 'pdf1')),
                    threading.Thread(target=process_pdf, args=(uploaded_files[1], results, 'pdf2'))
                ]
                
                for thread in threads:
                    thread.start()
                
                for thread in threads:
                    thread.join()
                
                st.session_state.results = results
                
            st.success("PDFs processed successfully!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display results
if st.session_state.results:
    st.subheader("Results Overview")
    st.write(f"Number of processed PDFs: {len(st.session_state.results)}")
    
    if len(st.session_state.results) == 2:
        st.subheader("Comparison Table")
        if 'comparison_table' not in st.session_state:
            with st.spinner("Generating comparison table..."):
                info1 = st.session_state.results['pdf1']['info_and_summary']
                info2 = st.session_state.results['pdf2']['info_and_summary']
                st.session_state.comparison_table = generate_difference_table(info1, info2)
        
        st.markdown(st.session_state.comparison_table)

       

        st.markdown("---")


    st.subheader("Individual Product Results")
    for pdf_key in ['pdf1', 'pdf2']:
     if pdf_key in st.session_state.results:
        result = st.session_state.results[pdf_key]
        
        # Extract product name using OpenAI
        product_name = extract_product_name_openai(result['info_and_summary'])
        
        with st.expander(f"Results for {product_name}"):
            info_lines = result['info_and_summary'].split('\n')
            for line in info_lines:
                line = line.strip()
                if line:
                    st.markdown(line)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label=f"Download English Text ({product_name})",
                    data=result['english_text'],
                    file_name=f"extracted_english_text_{product_name}.txt",
                    mime="text/plain",
                    key=f"download_english_{pdf_key}"
                )
            
            with col2:
                pdf_buffer = create_pdf(result['info_and_summary'])
                st.download_button(
                    label=f"Download Summary ({product_name})",
                    data=pdf_buffer,
                    file_name=f"product_info_and_summary_{product_name}.pdf",
                    mime="application/pdf",
                    key=f"download_summary_pdf_{pdf_key}"
                )

    st.markdown("---")
