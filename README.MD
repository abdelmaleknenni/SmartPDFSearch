 # 📄 Simple PDF Search Engine

A simple PDF search engine that uses **TF-IDF** and **Cosine Similarity** to find the most relevant paragraphs to a user query.

## 🚀 Project Idea
This program:
1. Reads all PDF files from a specified folder.
2. Extracts text and splits it into paragraphs.
3. Indexes the paragraphs and converts them into numerical vectors.
4. Searches for the closest matching paragraphs to the user query.

## 🛠 Requirements
Install the required libraries before running:

```bash
pip install pdfplumber scikit-learn