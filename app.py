import os
import streamlit as st
from dotenv import load_dotenv
from processor import extract_text
from embedder import create_faiss_index, load_faiss_index, get_similar_chunks
from llm_handler import generate_response, is_educational_with_llm
from deep_translator import GoogleTranslator
import asyncio
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
try:
    import speech_recognition as sr
except ImportError:
    sr = None
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load API key
load_dotenv()

UPLOAD_FOLDER = "data/uploaded_docs"
FAISS_INDEX_PATH = "data/faiss_index"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("🎓 Personalized Learning System - Educational Assistant")

# --- User Authentication ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Simple check, replace with real user validation in production
        if username and password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Please enter both username and password.")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.success("Logged out successfully.")

if not st.session_state.logged_in:
    login()
    st.stop()
else:
    st.sidebar.write(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        logout()
        st.stop()

# --- Session History ---
if 'history' not in st.session_state:
    st.session_state.history = []

# Show session history in sidebar
if st.session_state.history:
    st.sidebar.subheader("Session History")
    for i, item in enumerate(st.session_state.history[::-1]):
        st.sidebar.markdown(f"**Q:** {item['query']}")
        st.sidebar.markdown(f"**A:** {item['response']}")
        st.sidebar.markdown("---")

# 1. Choose interaction type
interaction_type = st.selectbox(
    "How would you like to interact?",
    ["Select", "Ask directly", "I have material to upload"]
)

vector_db = None
embedding_model = None

# 2. Upload and process material (only if user selects)
if interaction_type == "I have material to upload":
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])
    if uploaded_file:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ File uploaded successfully!")

        extracted_text = extract_text(file_path)
        if extracted_text.strip():
            num_chunks = create_faiss_index(extracted_text)
            st.info(f"🔍 {num_chunks} chunks created and stored in FAISS index.")
        else:
            st.error("❌ No readable text found in the uploaded document.")

    vector_db, embedding_model = load_faiss_index()
    if vector_db is None:
        st.warning("⚠️ FAISS index is missing. Please upload a document to continue.")

# 3. Help type selection
help_type = st.selectbox("What kind of help do you need?", ["Select", "Assignment", "Exam Questions", "Interview Questions", "Topic Explanations"])

# 4. Query box
query = st.text_area("✍️ Enter your educational question or topic")

# 5. Educational check (using LLM-based classification)
if st.button("Generate"):
    if help_type == "Select" or not query.strip() or interaction_type == "Select":
        st.warning("⚠️ Please select all required fields and enter a valid query.")
    elif not is_educational_with_llm(query):
        st.error("🚫 This assistant is only focused on **educational topics**. Please ask something related to learning or study.")
    else:
        # Ask directly
        if interaction_type == "Ask directly":
            # Empty FAISS usage — use just LLM
            response = generate_response(query, [], help_type)
        else:
            # Use FAISS
            retrieved_chunks = get_similar_chunks(query, vector_db, embedding_model)
            if retrieved_chunks:
                response = generate_response(query, retrieved_chunks, help_type)
            else:
                st.warning("⚠️ No relevant content found in the uploaded document.")
                response = None

        # Show response
        if response:
            st.subheader("📘 AI-Generated Content")
            # --- Multi-language Support ---
            language_options = ['English', 'Hindi', 'French', 'Spanish', 'German']
            language_codes = {'English': 'en', 'Hindi': 'hi', 'French': 'fr', 'Spanish': 'es', 'German': 'de'}
            selected_language = st.sidebar.selectbox('Select Output Language', language_options)

            def translate_text(text, target_lang):
                if target_lang == 'en':
                    return text
                return GoogleTranslator(source='auto', target=target_lang).translate(text)

            translated_response = translate_text(response, language_codes[selected_language])
            st.markdown(translated_response)
            # Save to session history
            st.session_state.history.append({
                'query': query,
                'response': translated_response,
                'feedback': None
            })
            # Feedback system
            feedback = st.radio("Rate this answer:", ["👍 Good", "👎 Needs Improvement"], key=f"feedback_{len(st.session_state.history)}")
            if feedback:
                st.session_state.history[-1]['feedback'] = feedback
                st.success(f"Feedback recorded: {feedback}")
            # Export/Download feature
            download_buffer = io.StringIO()
            download_buffer.write(f"Query: {query}\n\nResponse: {translated_response}")
            st.download_button(
                label="Download Response as TXT",
                data=download_buffer.getvalue(),
                file_name="response.txt",
                mime="text/plain"
            )

# --- Quiz/MCQ Generation ---
def generate_quiz(text, num_questions=5):
    # Simple placeholder using LLM; replace with your own logic or LLM prompt
    # You can use your LLM handler to generate MCQs from the text
    prompt = f"Generate {num_questions} multiple choice questions (with 4 options and answers) from the following text:\n{text}" 
    return generate_response(prompt, [], "Quiz")

if interaction_type == "I have material to upload" and uploaded_file:
    if st.button("Generate Quiz from Document"):
        quiz = generate_quiz(extracted_text)
        st.subheader("📝 Generated Quiz")
        st.markdown(quiz)

# --- Document Summarization ---
def summarize_document(text):
    prompt = f"Summarize the following document in simple terms:\n{text}"
    return generate_response(prompt, [], "Summary")

if interaction_type == "I have material to upload" and uploaded_file:
    if st.button("Summarize Document"):
        summary = summarize_document(extracted_text)
        st.subheader("📝 Document Summary")
        st.markdown(summary)
        # Export summary
        summary_buffer = io.StringIO()
        summary_buffer.write(summary)
        st.download_button(
            label="Download Summary as TXT",
            data=summary_buffer.getvalue(),
            file_name="summary.txt",
            mime="text/plain"
        )

    if st.button("Show Keyword Cloud"):
        # Generate word cloud from extracted text
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(extracted_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# --- Voice Input/Output ---
if sr:
    st.sidebar.subheader("Voice Input")
    if st.sidebar.button("Record Query"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Speak now...")
            audio = r.listen(source, timeout=5)
            try:
                voice_query = r.recognize_google(audio)
                st.success(f"You said: {voice_query}")
                query = voice_query
            except Exception as e:
                st.error(f"Could not recognize speech: {e}")
if pyttsx3:
    def speak_text(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    if st.button("Read Response Aloud") and response:
        speak_text(response)

# --- Admin Dashboard ---
def show_admin_dashboard():
    st.title("Admin Dashboard")
    st.subheader("Usage Statistics")
    st.write(f"Total queries this session: {len(st.session_state.history)}")
    feedback_counts = {"👍 Good": 0, "👎 Needs Improvement": 0}
    for item in st.session_state.history:
        if item.get('feedback'):
            feedback_counts[item['feedback']] += 1
    st.write(f"Feedback Summary: {feedback_counts}")
    st.subheader("User Management (Demo)")
    st.write(f"Current user: {st.session_state.username}")
    # Add more admin features as needed

if st.session_state.username == "admin":
    show_admin_dashboard()
    st.stop()
