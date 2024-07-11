import streamlit as st
import PyPDF2
from deep_translator import GoogleTranslator
import openai
from textwrap import wrap
import concurrent.futures
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def translate_text(text, src='es', dest='en'):
    translator = GoogleTranslator(source=src, target=dest)
    chunk_size = 4000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []
    for chunk in chunks:
        try:
            translated_chunks.append(translator.translate(chunk))
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            translated_chunks.append(chunk)
    return ' '.join(translated_chunks)

def analyze_text_with_gpt(text):
    prompt = f"""Analyze the following part of a pharmaceutical product information and provide a partial structured summary:
    # Partial Analysis of Pharmaceutical Product
    [Provide relevant information for any of the following categories that are present in this text chunk.
    If a category is not addressed in this chunk, omit it from the summary.]
    1. **Product Name**
       [which is always 2-3 words]
    2. **Brief Description**
    3. **Composition (Active ingredients)**
    4. **Excipients with Known Effects**
    5. **Dosage Form**
    6. **Posology**(include info based on body surface area if applicable)
    7. **Warnings**
    8. **Shelf Life**(include storage conditions, stability, and instructions on specific storage after reconstitution in easily understandable way. Include exact durations, temperatures, and any special handling instructions. If information is not available, clearly state "Information not provided")
    9. **Indications**
    10. **Other essential information**(include other information which are important)
    Text chunk: {text}
    Note: Provide the output with headings in bold and normal text size for the content. Ensure that the Shelf Life information is detailed and complete.
    """
   
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes pharmaceutical product and provide exact details."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_final_summary(summaries):
    combined_summary = "\n\n".join(summaries)
    final_prompt = f"""
    Based on the following combined partial analyses of a pharmaceutical product, provide a final structured summary:
    {combined_summary}
    Please format the final summary as follows:
    **Analysis of Pharmaceutical Product**
    1. **Product Name**
    [Product Name which is just 2-3 words]
    2. **Brief Description**
    [2-3 sentence description of the product]
    3. **Composition**
    - Active ingredient(s):
      - [Ingredient 1]
      - [Ingredient 2]
      - ...
    4. **Excipients with Known Effects**
    - [Excipient 1]
    - [Excipient 2]
    - ...
    5. **Dosage Form**
    [Dosage form]
    6. **Posology**
    [Brief summary of dosage instructions]
    7. **Warnings**
    - [Warning 1]
    - [Warning 2]
    - ...
    8. **Shelf Life**
    [Provide detailed shelf life information, including storage conditions, stability, and any special instructions]
    
    9. **Indications**
    [List of indications]
    
    10. **Other Essential Information**
    - [Information 1]
    - [Information 2]
    - ...
    11. **Overall Summary**
    [A concise summary of the key points from the text]
    Note: Provide the output with headings in bold and normal text size for the content. Ensure that the Shelf Life information is detailed and complete.
    """
   
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes pharmaceutical product information."},
            {"role": "user", "content": final_prompt}
        ]
    )
    return response.choices[0].message.content

def create_difference_table(info1, info2):
    prompt = f"""
    Create a detailed difference table for the following two pharmaceutical products. Use the product names as the top heading. For each key information heading, provide a concise comparison of the two products.
   
    Product 1 Information:
    {info1}
   
    Product 2 Information:
    {info2}
   
    Format the output as a markdown table with the following columns:
    - **Key Information**
    - **Product 1**
    - **Product 2**

    Ensure that you include a detailed comparison of the Shelf Life information for both products in the table.

    After the table, provide the following sections as regular text (not in table format):
    **Advantages and Competitive Edge:**
    - For Product 1, provide a detailed list of advantages and competitive edges. Include specific information about:
      - Effectiveness for its intended use
      - Unique features or formulation
      - Dosage convenience
      - Side effect profile
      - Shelf life and storage advantages
      - Any other relevant factors that set it apart
    - Repeat the same detailed analysis for Product 2

    **Overall Conclusion:**
    - Provide an overall evaluation comparing both products.
    - Discuss potential scenarios or patient profiles where one product might be preferred over the other.
    - Summarize the key differences and their implications for treatment, including any differences in shelf life or storage requirements.
    - Conclude with a balanced perspective, acknowledging that the final choice should be made in consultation with healthcare professionals.

    Ensure all headings are in bold letters and the output is clear, concise, and informative.
    """
   
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that compares pharmaceutical products."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def process_pdf(uploaded_file):
    text = extract_text_from_pdf(uploaded_file)
    translated_text = translate_text(text)
   
    chunk_size = 2000
    chunks = wrap(translated_text, chunk_size)
   
    partial_analyses = []
    for chunk in chunks:
        partial_analyses.append(analyze_text_with_gpt(chunk))
   
    final_summary = generate_final_summary(partial_analyses)
   
    return final_summary, translated_text

def extract_product_name(summary):
    lines = summary.split('\n')
    for line in lines:
        if line.startswith('1. **Product Name**'):
            return ' '.join(line.split()[3:]).strip()
    return "Product Name Not Found"

def process_pdfs_in_parallel(uploaded_files):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_pdf, pdf): pdf.name for pdf in uploaded_files}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            pdf_name = futures[future]
            try:
                summary, translated_text = future.result()
                results[pdf_name] = (summary, translated_text)
            except Exception as e:
                st.error(f"Error processing PDF {pdf_name}: {e}")
        return results

def answer_query(query, summary1, summary2):
    prompt = f"""
    You are an assistant knowledgeable in pharmaceutical products. Answer the following question based on the provided summaries of two pharmaceutical products:
   
    Question: {query}
   
    Summary of Product 1:
    {summary1}
   
    Summary of Product 2:
    {summary2}
   
    Provide a concise and informative response.
    """
   
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant knowledgeable in pharmaceutical products."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def main():
    st.title("SMPC summarization in English")

    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    uploaded_files = st.file_uploader("Upload two Spanish PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and len(uploaded_files) == 2:
        if st.button("Translate and Analyze"):
            with st.spinner("Processing PDFs..."):
                results = process_pdfs_in_parallel(uploaded_files)
                summary1, translated_text1 = results[uploaded_files[0].name]
                summary2, translated_text2 = results[uploaded_files[1].name]
                product_name1 = extract_product_name(summary1)
                product_name2 = extract_product_name(summary2)
                st.session_state.summary1 = summary1
                st.session_state.translated_text1 = translated_text1
                st.session_state.product_name1 = product_name1
                st.session_state.summary2 = summary2
                st.session_state.translated_text2 = translated_text2
                st.session_state.product_name2 = product_name2

            with st.spinner("Generating difference table and comparison..."):
                difference_table = create_difference_table(st.session_state.summary1, st.session_state.summary2)
                st.session_state.difference_table = difference_table

            st.session_state.analysis_complete = True

    if st.session_state.analysis_complete:
        st.subheader("Difference Table and Comparison")
        st.markdown(st.session_state.difference_table)

        st.subheader(st.session_state.product_name1)
        st.markdown(st.session_state.summary1)

        st.subheader(st.session_state.product_name2)
        st.markdown(st.session_state.summary2)

        col1, col2, col3, col4 = st.columns(4)
       
        with col1:
            st.download_button(
                label="Download Summary 1",
                data=st.session_state.summary1,
                file_name="summary1.txt",
                mime="text/plain"
            )
       
        with col2:
            st.download_button(
                label="Download Summary 2",
                data=st.session_state.summary2,
                file_name="summary2.txt",
                mime="text/plain"
            )

        with col3:
            st.download_button(
                label="Download Translated Text 1",
                data=st.session_state.translated_text1,
                file_name="translated_text1.txt",
                mime="text/plain"
            )

        with col4:
            st.download_button(
                label="Download Translated Text 2",
                data=st.session_state.translated_text2,
                file_name="translated_text2.txt",
                mime="text/plain"
            )
       
        st.subheader("Chat assistant")
        user_query = st.text_input("Enter your question about the products")
       
        if user_query:
            with st.spinner("Generating answer..."):
                answer = answer_query(user_query, st.session_state.summary1, st.session_state.summary2)
                st.markdown(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()
