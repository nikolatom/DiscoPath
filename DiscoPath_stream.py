import time
import requests
import pywikipathways
import xml.etree.ElementTree as ET
import streamlit as st
from openai import OpenAI
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import zipfile
import uuid
import shutil

# Define your OpenAI API key
client = OpenAI(api_key="")

output_lock = threading.Lock()  # Lock for thread-safe file writing
error_lock = threading.Lock()  # Lock for thread-safe error collection
log_lock = threading.Lock()  # Lock for thread-safe logging
errors = []  # List to collect errors
query_logged = False  # Flag to track if the query has been logged

def create_output_dir(base_dir="tmp"):
    """Create a unique output directory under the base directory."""
    unique_id = str(uuid.uuid4())
    output_dir = os.path.join(base_dir, unique_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def import_text_file_to_dataframe(uploaded_file):
    """Reads a text file from an uploaded file object and returns its contents as a pandas DataFrame."""
    try:
        content = uploaded_file.read().decode("utf-8")
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        data_frame = pd.DataFrame(lines, columns=['Gene'])
        return data_frame
    except Exception as e:
        with error_lock:
            errors.append(f"An error occurred while reading the file: {e}")
        return None

def find_pathways_by_text(gene_symbol):
    """Retrieve pathways associated with a gene symbol from WikiPathways."""
    url = f"https://webservice.wikipathways.org/findPathwaysByText?query={gene_symbol}&species=Homo sapiens&format=xml"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        with error_lock:
            errors.append(f"Error fetching data from WikiPathways for gene: {gene_symbol}")
        return None

def parse_pathways(xml_data):
    """Parse XML data to extract pathways."""
    if xml_data is None:
        return []
    ns = {'ns1': 'http://www.wso2.org/php/xsd', 'ns2': 'http://www.wikipathways.org/webservice'}
    root = ET.fromstring(xml_data)
    pathways = []
    for result in root.findall('ns1:result', ns):
        id_elem = result.find('ns2:id', ns)
        name_elem = result.find('ns2:name', ns)
        if id_elem is not None and name_elem is not None:
            pathways.append({'id': id_elem.text, 'name': name_elem.text})
    return pathways

def extract_ids(pathways_id_names):
    """Extract the 'id' values from a list of dictionaries representing pathways."""
    return [pathway['id'] for pathway in pathways_id_names]

def fetch_pathway_details(pathway_id):
    """Fetch pathway details for a given pathway ID."""
    try:
        pathway_info = pywikipathways.get_pathway(pathway_id)
        return pathway_info
    except Exception as e:
        with error_lock:
            errors.append(f"An error occurred fetching pathway details for ID {pathway_id}: {e}")
        return None

def remove_lines(pathway_info):
    """Remove lines from the XML data that start with any of the specified patterns."""
    patterns = [
        r'<Attribute Key', r'<Graphics', r'<DataNode', r'</DataNode', r'<Point', r'<Anchor', r'</Graphic', r'<Graphic',
        r'<Interaction', r'</Interaction', r'ArrowHead', r'RelX', r'X', r'<Xref', r'<BiopaxRef', r'</Group', r'<Group',
        r'<bp:DB', r'</bp:PublicationXref', r'<Label', r'</Label', r'</Shape', r'<Shape', r'<InfoBox'
    ]
    pattern = re.compile('|'.join([f'^{p}' for p in patterns]))
    filtered_data = '\n'.join(line for line in pathway_info.split('\n') if not pattern.match(line.strip()))
    return filtered_data

def lipid_narrative_analysis(gene_symbol, pathway, model):
    """Generate narrative analysis using OpenAI GPT model."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant, a professional in biological pathway analysis. Make sure that all the pathways or interest are included and formatted identically."},
        {"role": "user", "content": f"Provide detailed analysis for the gene symbol '{gene_symbol}' based on '{pathway}' that includes its associated pathways, diseases, and any relevant publications with PMIDs. Highlight the significance of these pathways in human biology and disease, and mention any notable findings from the publications."}
    ]
    try:
        response = client.chat.completions.create(model=model, messages=messages, stream=True, temperature=0)
        narrative_summary = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                narrative_summary += chunk.choices[0].delta.content
        return narrative_summary.strip()
    except Exception as e:
        error_message = f"An error occurred during gene narrative analysis for gene {gene_symbol}: {e}"
        with error_lock:
            errors.append(error_message)
        return None

def detailed_lipid_pathways_table(gene_symbol, narrative_analysis_results, model):
    """Generate a detailed table of pathways using OpenAI GPT model."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant, a professional in biological pathway analysis providing a consistent output format. Make sure that all the pathways or interest are included and formatted identically."},
        {"role": "user", "content": f"Based on the pathways information provided for '{gene_symbol}', create a table describing the pathways in details. Include the pathway name, ID, and its role in human biology and disease, and publications with PMID:\n{narrative_analysis_results}. Provide .txt output format"}
    ]
    try:
        response = client.chat.completions.create(model=model, messages=messages, stream=True, temperature=0)
        table_summary = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                table_summary += chunk.choices[0].delta.content
        return table_summary.strip()
    except Exception as e:
        error_message = f"An error occurred during detailed gene pathways table generation for gene {gene_symbol}: {e}"
        with error_lock:
            errors.append(error_message)
        return None

def save_results_to_file(file_path, gene_symbol, detailed_pathways_table):
    """Save the results to a file and return the content written."""
    markdown_content = f"Results for gene: {gene_symbol}\n{detailed_pathways_table}\n" + "=" * 80 + "\n"
    with output_lock:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(markdown_content)
    return markdown_content

def save_relevant_pathways_to_file(file_path, relevant_pathways):
    """Save the relevant pathways to a file."""
    with output_lock:
        with open(file_path, 'a', encoding='utf-8') as file:
            for pathway in relevant_pathways:
                file.write(f"{pathway['Gene Symbol']},{pathway['Pathway Name']},{pathway['Pathway ID']}\n")

def log_message(log_file, message):
    """Log a message to the specified log file."""
    with log_lock:
        with open(log_file, 'a', encoding='utf-8') as file:
            file.write(message + '\n')

def check_pathways_relevance(lipid_symbol, pathways, query, model, log_file):
    """Check if each pathway in the list is relevant to the specified query using AI and return details including the gene symbol in a DataFrame."""
    relevant_pathways = []
    for pathway in pathways:
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to precisely decide if the query is fitting the provided pathways."},
            {"role": "user", "content": f"Is the pathway '{pathway['name']}' {query}?"}
        ]
        try:
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=500, temperature=0)
            if response.choices and response.choices[0].message.content:
                answer = response.choices[0].message.content.strip().lower()
                if "yes" in answer or "true" in answer:
                    relevant_pathways.append({'Gene Symbol': lipid_symbol, 'Pathway Name': pathway['name'], 'Pathway ID': pathway['id']})
                    log_message(log_file, f"Pathway '{pathway['name']}' is relevant to the gene '{lipid_symbol}' regarding the user's query.")
                else:
                    log_message(log_file, f"Pathway '{pathway['name']}' is not relevant to the gene '{lipid_symbol}' regarding the user's query.")
            else:
                error_message = f"No content received from AI response for pathway {pathway['name']}."
                with error_lock:
                    errors.append(error_message)
        except Exception as e:
            error_message = f"An error occurred during AI query processing for pathway {pathway['name']}: {e}"
            with error_lock:
                errors.append(error_message)
    return pd.DataFrame(relevant_pathways)

def process_gene(gene_symbol, output_dir, query, pathways_filtering, detailed_annotations, model, log_file):
    """Process a single gene symbol."""
    log_message(log_file, f"Processing gene: {gene_symbol}")
    global query_logged
    if not query_logged:
        log_message(log_file, f"Query: {query}")
        query_logged = True
    xml_data = find_pathways_by_text(gene_symbol)
    pathway_id_names = parse_pathways(xml_data)
    if not pathway_id_names:
        error_message = f"No pathways found for gene: {gene_symbol}"
        with error_lock:
            errors.append(error_message)
        return
    relevant_pathways_file = os.path.join(output_dir, f"relevant_pathways_{gene_symbol}.txt")
    detailed_pathways_file = os.path.join(output_dir, f"detailed_pathways_{gene_symbol}.txt")

    if pathways_filtering:
        relevant_pathways_df = check_pathways_relevance(gene_symbol, pathway_id_names, query, model, log_file)
        if relevant_pathways_df.empty:
            error_message = f"No relevant pathways found for gene: {gene_symbol}"
            with error_lock:
                errors.append(error_message)
            return
        save_relevant_pathways_to_file(relevant_pathways_file, relevant_pathways_df.to_dict('records'))
        pathway_ids = relevant_pathways_df['Pathway ID'].tolist()
    else:
        pathway_ids = extract_ids(pathway_id_names)

    all_pathway_details = {}

    for pathway_id in pathway_ids:
        pathway_info = fetch_pathway_details(pathway_id)
        if pathway_info:
            pathway_info = remove_lines(pathway_info)
        all_pathway_details[pathway_id] = pathway_info

    if detailed_annotations:
        narrative_analysis_results = {}
        for pathway_id, pathway_info in all_pathway_details.items():
            if pathway_info:
                narrative_analysis = lipid_narrative_analysis(gene_symbol, pathway_info, model)
                narrative_analysis_results[pathway_id] = narrative_analysis

        detailed_pathways_table = detailed_lipid_pathways_table(gene_symbol, narrative_analysis_results, model)
        if detailed_pathways_table:
            save_results_to_file(detailed_pathways_file, gene_symbol, detailed_pathways_table)
        else:
            error_message = f"Failed to generate detailed pathways table for gene: {gene_symbol}"
            with error_lock:
                errors.append(error_message)

def concatenate_files(output_dir, output_filename, pattern):
    """Concatenate files matching the pattern in the output directory into a single file."""
    concatenated_file_path = os.path.join(output_dir, output_filename)
    with open(concatenated_file_path, 'w', encoding='utf-8') as outfile:
        for root, _, files in os.walk(output_dir):
            for file in sorted(files):
                if re.match(pattern, file):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as infile:
                        shutil.copyfileobj(infile, outfile)
    return concatenated_file_path

def concatenate_first_column(output_dir, output_filename, pattern):
    """Concatenate the first column of files matching the pattern in the output directory into a single file."""
    concatenated_file_path = os.path.join(output_dir, output_filename)
    seen_genes = set()
    with open(concatenated_file_path, 'w', encoding='utf-8') as outfile:
        for root, _, files in os.walk(output_dir):
            for file in sorted(files):
                if re.match(pattern, file):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as infile:
                        for line in infile:
                            gene_symbol = line.split(',')[0]
                            if gene_symbol not in seen_genes:
                                outfile.write(gene_symbol + '\n')
                                seen_genes.add(gene_symbol)
    return concatenated_file_path

def zip_output_dir(output_dir, zip_filename):
    """Create a ZIP file of the output directory."""
    zip_file_path = os.path.join(output_dir, zip_filename)
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.txt') or file.endswith('.log'):
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))
    return zip_file_path

def main():
    st.title("DiscoPath - connect features to WikiPathways by AI")

    gene_file_path = st.file_uploader("Upload Gene File", type="txt")

    pathways_filtering = st.checkbox("Enable Pathway Filtering", value=True)
    query = ""
    if pathways_filtering:
        query = st.text_input("Enter Query for Pathway Filtering", "", help="Example: cholesterol regulation, lipidomics, breast cancer, ovarian cancer")

    detailed_annotations = st.checkbox("Enable Detailed Pathway Annotations (this might take a while and be expensive)", value=False)

    model = st.selectbox("Select GPT Model", ["gpt-4o", "gpt-3.5-turbo", "gpt-4"], index=1)

    if gene_file_path:
        output_dir = create_output_dir()
        zip_filename = st.text_input("Enter the filename for the results ZIP file", "results.zip")
        log_file = os.path.join(output_dir, "normal_output.log")
        error_log_file = os.path.join(output_dir, "error_output.log")
        
        if st.button("Analyze"):
            analysis_placeholder = st.empty()  # Placeholder for analysis running message
            analysis_placeholder.text("Analysis running...")

            gene_df = import_text_file_to_dataframe(gene_file_path)
            if gene_df is not None:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(process_gene, gene_symbol, output_dir, query, pathways_filtering, detailed_annotations, model, log_file): gene_symbol for gene_symbol in gene_df['Gene']}
                    for future in as_completed(futures):
                        gene_symbol = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            with error_lock:
                                errors.append(f"An error occurred while processing gene {gene_symbol}: {e}")

                analysis_placeholder.empty()  # Clear the analysis running message
                st.success("Analysis complete!")

                # Concatenate files
                concatenate_files(output_dir, "all_relevant_pathways.txt", r"relevant_pathways_.*\.txt")
                concatenate_first_column(output_dir, "all_relevant_genes.txt", r"relevant_pathways_.*\.txt")
                if detailed_annotations:
                    concatenate_files(output_dir, "all_detailed_pathways.txt", r"detailed_pathways_.*\.txt")

                # Write collected errors to the error log file
                with open(error_log_file, 'w', encoding='utf-8') as error_log:
                    for error in errors:
                        error_log.write(error + '\n')

                # Create ZIP file for download
                zip_file_path = zip_output_dir(output_dir, zip_filename)
                with open(zip_file_path, "rb") as fp:
                    st.download_button(
                        label="Download Results",
                        data=fp,
                        file_name=os.path.basename(zip_file_path),
                        mime="application/zip"
                    )

                # Display collected errors, if any
                if errors:
                    for error in errors:
                        st.error(error)
            else:
                st.error("Error reading gene file.")

if __name__ == "__main__":
    main()
