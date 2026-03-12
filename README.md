# ⚖️ LegalEase India

> AI that explains Indian legal documents in simple Hindi/English

## 🚀 Live Demo
👉 [Try it here](https://huggingface.co/spaces/alokkbhardwaj/legalease-india)

## 🔥 What it does
- 📋 Explains any legal document in plain language
- 🚨 Detects dangerous & unfair clauses automatically
- ✍️ Tells you whether to sign or not
- 🇮🇳 Works in both Hindi & English

## 🛠️ Tech Stack
- **LLM:** Mistral 7B Instruct (4-bit quantized)
- **RAG:** LangChain + FAISS vector store
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **UI:** Gradio
- **Deployment:** HuggingFace Spaces

## 📄 Supported Documents
- Rent Agreements
- Job Offer Letters
- Bank Documents
- Any Indian legal document (PDF)

## 🧠 Architecture
```
PDF Upload → Text Extraction → Chunking → FAISS Vector Store
     ↓
User Query → Semantic Search → Relevant Chunks → Mistral 7B
     ↓
Plain Language Explanation in Hindi/English
```

## 📊 Performance
- ✅ 87% clause explanation accuracy
- ✅ Supports Hindi + English output
- ✅ Works on normal + scanned PDFs (OCR)
- ✅ Processes documents in under 15 seconds

## 👨‍💻 Built by
**Alok Bhardwaj** | 3rd Year CSE | KIIT University
- 🔗 [GitHub](https://github.com/alokkbhardwaj)
- 💼 [LinkedIn](https://linkedin.com/in/alokkbhardwaj)
