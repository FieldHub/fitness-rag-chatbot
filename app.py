import streamlit as st
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Fitness RAG Assistant", page_icon="🏋️‍♂️")
st.title("🏋️‍♂️ ผู้ช่วยอัจฉริยะด้านการออกกำลังกาย")

# ดึง API Key
if "GEMINI_API_KEY" not in st.secrets:
    st.error("ไม่พบ API Key กรุณาตั้งค่าใน .streamlit/secrets.toml หรือ Streamlit Secrets")
    st.stop()
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Cache ไว้ไม่ให้โหลดใหม่ทุกครั้งที่พิมพ์แชต
@st.cache_resource
def load_and_process_documents():
    documents = []
    for file_path in glob.glob("data/*"):
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
        except Exception as e:
            st.warning(f"โหลดไฟล์ {file_path} ไม่สำเร็จ: {e}")
            
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

with st.spinner("กำลังโหลดคลังความรู้ฟิตเนส..."):
    vector_store = load_and_process_documents()
    if vector_store is None:
        st.error("ไม่พบเอกสารในโฟลเดอร์ data/ กรุณาเพิ่มเอกสารก่อนใช้งาน")
        st.stop()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ตั้งค่า LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=GEMINI_API_KEY, temperature=0.1)

# Prompt บังคับเงื่อนไข
template = (
    "คุณคือผู้ช่วยเชี่ยวชาญด้านการออกกำลังกาย จงตอบคำถามโดยใช้ข้อมูลจาก Context ที่ให้มาเท่านั้น "
    "หากใน Context ไม่มีข้อมูลที่เกี่ยวข้อง ให้ตอบกลับมาตรงๆ ว่า 'ไม่พบข้อมูล' ห้ามแต่งเนื้อหาขึ้นมาเองเด็ดขาด "
    "และเมื่อตอบคำถามสำเร็จ ให้ระบุแหล่งที่มา (Source) ที่ใช้ในการตอบด้วยเสมอ\n\n"
    "Context:\n{context}\n\n"
    "คำถาม: {question}\n\n"
    "คำตอบ:"
)
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# สร้าง Chain แบบ LCEL (LangChain Expression Language) ที่เสถียรที่สุด
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# UI แชตบอต
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "สวัสดีครับ! ผมคือผู้ช่วยด้านฟิตเนส มีคำถามเรื่องท่าออกกำลังกายหรือโภชนาการ ถามมาได้เลยครับ"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("พิมพ์คำถามเกี่ยวกับการออกกำลังกาย..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    with st.spinner("กำลังวิเคราะห์ข้อมูล..."):
        try:
            # ดึงเอกสารอ้างอิงมาแสดงผลร่วมด้วย
            retrieved_docs = retriever.invoke(user_input)
            answer = rag_chain.invoke(user_input)
            
            sources = set([doc.metadata.get("source", "ไม่ทราบแหล่งที่มา") for doc in retrieved_docs])
            source_text = "\n\n**อ้างอิงจาก:** " + ", ".join(sources)
            
            if "ไม่พบข้อมูล" in answer:
                full_answer = answer
            else:
                full_answer = answer + source_text
                
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            st.chat_message("assistant").write(full_answer)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")