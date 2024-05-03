# 2024-AI-Challenge-LawScript
Submission source code for AI Challenge 2024 for team LawScript - An AI Legal Search Engine.

# LawScript AI

LawScript AI is a Streamlit-based application that utilizes Retrieval Augmented Generation to answer legal queries. The sample data being utilised for search from current_data folder is already preprocessed using data science techniques (this has been done separately, so the project focuses on our core value proposition).

## Features

- Load and search through embeddings of legal documents.
- Utilize OpenAI's powerful models to generate informed responses based on the content of similar legal documents.
- Streamlit interface for easy interaction.

## Getting Started

### Prerequisites

Before you can run LawScript AI, you need to install the necessary Python packages. This project requires Python 3.6 or higher.

### Installation

1. **Clone the repository**

    ```bash
    git clone git@github.com:wasifhassan101/2024-AI-Challenge-LawScript.git
    cd your-repository-directory
    ```


2. **Set up a virtual environment** (optional, but recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```


3. **Install the dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your OpenAI API key**

    You need an API key from OpenAI to use their models. Once you have your API key, you can set it directly in the `lawscript.py` or preferably, use an environment variable to keep it secure.

    Set your api_key directly in lawscript.py:

    ```bash
    OPENAI_API_KEY='your-api-key-here'
    ```


### Running the Application

To run the application, use the following command:

```bash
streamlit run lawscript.py
```

### Utilising the Data
There is preprocessed data in the reserve_data folder. To make the setup quick, only the Constitution and Criminal Code are in the current_data. There is one more legal Act on FIA in the reserve_data directory. To use more data for the current run, move the desired .csv file from reserve_data to current_data folder.
