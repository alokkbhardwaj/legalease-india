import gradio as gr
import pdfplumber
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# ─── Load Models ───────────────────────────────────────────────
print("⏳ Models load ho rahe hain...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
    do_sample=True,
    repetition_penalty=1.1
)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
print("✅ Models ready!")

# ─── Sample Document ────────────────────────────────────────────
SAMPLE_TEXT = """
RENTAL AGREEMENT

This Rental Agreement is made on 1st January 2025 between:
LANDLORD: Ramesh Kumar Sharma, residing at 45, MG Road, Bhubaneswar
TENANT: Rahul Singh, student at KIIT University

CLAUSE 1 - RENT AND PAYMENT
The monthly rent shall be Rs. 8,000/- (Eight Thousand Rupees only).
Rent must be paid by 5th of every month. Late payment will attract
a penalty of Rs. 500/- per day after the due date.

CLAUSE 2 - SECURITY DEPOSIT
The tenant shall pay a security deposit of Rs. 40,000/- which is
NON-REFUNDABLE under any circumstances including early termination.

CLAUSE 3 - LOCK-IN PERIOD
The tenant must stay for minimum 11 months. Early exit will result
in forfeiture of entire security deposit plus 2 months penalty rent.

CLAUSE 4 - MAINTENANCE
All maintenance and repair costs shall be borne entirely by the tenant,
including structural repairs, plumbing, and electrical work.

CLAUSE 5 - VISITORS
No visitors allowed after 9 PM. Violation will result in immediate
termination of agreement with no refund of security deposit.

CLAUSE 6 - TERMINATION
Landlord reserves the right to terminate this agreement with 24 hours
notice without giving any reason whatsoever.
"""

# ─── Core Function ──────────────────────────────────────────────
def analyze(pdf_file, question, language):
    if pdf_file is not None:
        try:
            with pdfplumber.open(pdf_file.name) as pdf:
                text = "\n".join([p.extract_text() or "" for p in pdf.pages])
            if len(text.strip()) < 50:
                text = SAMPLE_TEXT
        except:
            text = SAMPLE_TEXT
    else:
        text = SAMPLE_TEXT

    chunks = splitter.split_text(text)
    vs = FAISS.from_texts(chunks, embeddings)
    docs = vs.as_retriever(search_kwargs={"k": 3}).invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    lang_inst = (
        "Apna jawab simple Hindi mein do. Legal jargon avoid karo."
        if language == "Hindi"
        else "Answer in simple English. Avoid legal jargon. Be direct and practical."
    )

    prompt = f"""<s>[INST] You are LegalEase India, an AI that helps Indians understand legal documents simply.
{lang_inst}
Flag anything dangerous or unfair clearly.

Document excerpts:
{context}

Question: {question} [/INST]"""

    result = llm(prompt)[0]["generated_text"].split("[/INST]")[-1].strip()
    return result

# ─── Gradio UI ──────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="⚖️ LegalEase India") as app:
    gr.Markdown("""
    # ⚖️ LegalEase India
    ### AI jo aapke legal documents simple Hindi/English mein explain kare
    *Rent agreements, offer letters, bank documents — sab kuch!*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="📄 PDF Upload karo",
                file_types=[".pdf"]
            )
            gr.Markdown("*PDF nahi hai? Sample rent agreement pe test hoga!*")
            language = gr.Radio(
                ["English", "Hindi"],
                value="English",
                label="🌐 Language / भाषा"
            )
            with gr.Row():
                btn1 = gr.Button("📋 Summary", variant="primary", size="lg")
            with gr.Row():
                btn2 = gr.Button("🚨 Red Flags", variant="stop", size="lg")
            with gr.Row():
                btn3 = gr.Button("✍️ Sign or Not?", variant="secondary", size="lg")

        with gr.Column(scale=2):
            output = gr.Textbox(
                label="🤖 LegalEase India says:",
                lines=18,
                show_copy_button=True,
                placeholder="Upar se koi bhi button click karo..."
            )

    gr.Markdown("""
    ---
    ⚠️ *Ye AI legal advice nahi hai. Kisi bhi important document ke liye real lawyer se consult karo.*
    
    Built with ❤️ by Alok Bhardwaj | KIIT University | [GitHub](https://github.com/alokkbhardwaj)
    """)

    btn1.click(
        lambda f, l: analyze(f, "Explain this entire document in simple terms. What are the key points I must know?", l),
        inputs=[pdf_input, language], outputs=output
    )
    btn2.click(
        lambda f, l: analyze(f, "What are the most dangerous, unfair or risky clauses? List each one clearly.", l),
        inputs=[pdf_input, language], outputs=output
    )
    btn3.click(
        lambda f, l: analyze(f, "Should I sign this document? Give me a clear YES or NO with specific reasons.", l),
        inputs=[pdf_input, language], outputs=output
    )

app.launch()
