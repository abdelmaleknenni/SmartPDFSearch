import os
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_into_paragraphs(text):
    return [para.strip() for para in text.split("\n\n") if len(para.strip()) > 30]

def index_documents(folder_path):
    paragraphs = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    for file in files:
        file_path = os.path.join(folder_path, file)
        text = extract_text_from_pdf(file_path)
        paras = split_into_paragraphs(text)
        paragraphs.extend(paras)
    return paragraphs

def search(query, paragraphs, top_k=5):
    vectorizer = TfidfVectorizer().fit(paragraphs + [query])
    para_vectors = vectorizer.transform(paragraphs)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, para_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(paragraphs[i], similarities[i]) for i in top_indices]

if __name__ == "__main__":
    print("🔎 Simple Document Search Engine (TF-IDF)")
    folder = input("📂 Enter folder path containing PDF files:\n> ").strip()
    print("📄 Indexing documents...")
    paragraphs = index_documents(folder)
    print(f"✅ Indexing complete. {len(paragraphs)} paragraphs indexed. You can now search.")
    
    while True:
        query = input("\nType your question (or 'exit' to quit):\n> ").strip()
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
        results = search(query, paragraphs)
        print("\n🔍 Top results:\n" + "="*50)
        for i, (para, score) in enumerate(results, 1):
            print(f"Result {i} - Score: {score:.3f}")
            print("-" * 50)
            # عرض الفقرة مع تقطيعها إلى أسطر قصيرة 80 حرف
            words = para.split()
            line_length = 0
            line_words = []
            for w in words:
                line_words.append(w)
                line_length += len(w) + 1
                if line_length > 80:
                    print(" ".join(line_words))
                    line_words = []
                    line_length = 0
            if line_words:
                print(" ".join(line_words))
            print("="*50)
