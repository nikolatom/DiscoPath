
# DiscoPath - Connect Features to WikiPathways by AI

## Overview
DiscoPath is a Streamlit-based application with GUI designed to connect genetic information to pathways in WikiPathways. By leveraging powerful APIs like OpenAI and Wikipathways, alongside local analysis tools, DiscoPath facilitates comprehensive studies of feature-related pathways and their implications in human biology.

## Features
- **Dynamic Pathway Retrieval**: Utilizes WikiPathways API to dynamically fetch pathways related to specific feature symbols (gene/protein/lipid/metabolite/etc).
- **Pathway filtering by AI**: Employs OpenAI's GPT models to select pathways and features fitting the user provided query.
- **Advanced Narrative Analysis by AI**: Employs OpenAI's GPT models to generate detailed narrative analyses, including associated diseases and publications.
- **Concurrent Processing**: Supports multi-threaded processing for simultaneous analyses of multiple genes, enhancing performance.

## Installation Instructions

### Clone the Repository
```bash
git clone https://github.com/nikolatom/DiscoPath.git
cd DiscoPath
```

### Set Up Python Environment
You can set up the environment using either venv or conda:


#### Using Conda
1. **Create a Conda environment**:
   ```bash
   conda create --prefix ./env_DiscoPath python=3.12 
   ```

2. **Activate the Conda environment**:
   ```bash
   conda activate ./env_DiscoPath
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

#### Using venv
1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


## Configuration
Configure your OpenAI API key in the DiscoPath.py to enable AI-driven functionalities:
```python
client = OpenAI(api_key="your_openai_api_key")
```

## Usage
Launch the application by running:
```bash
streamlit run DiscoPath_stream.py
```
Navigate to http://localhost:8501 in your web browser. Upload a text file containing gene symbols, specify your desired settings, eventually provide the query for pathways filtering and initiate the analysis. Results will be exported into the output directory.

## Contributing
We welcome contributions! Please fork the repository and submit pull requests, or open issues for bugs and feature requests.

## License
DiscoPath is released under the MIT License. See the LICENSE file in the repository for more details.
