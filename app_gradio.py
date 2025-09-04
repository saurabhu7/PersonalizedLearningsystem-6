import gradio as gr
from processor import extract_text
from embedder import create_faiss_index, load_faiss_index, get_similar_chunks
from llm_handler import generate_response, is_educational_with_llm

def answer_question(query, help_type):
    if help_type == "Select" or not query.strip():
        return "⚠️ Please select a help type and enter a valid query."
    elif not is_educational_with_llm(query):
        return "🚫 This assistant is only focused on educational topics."
    else:
        # For now, just use LLM (no FAISS)
        response = generate_response(query, [], help_type)
        return response

iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(label="Enter your educational question or topic"),
        gr.Radio(["Select", "Assignment", "Exam Questions", "Interview Questions", "Topic Explanations"], label="What kind of help do you need?")
    ],
    outputs="text",
    title="🎓 Personalized Learning System - Educational Assistant",
    description="Ask educational questions and get AI-powered answers instantly."
)

if __name__ == "__main__":
    iface.launch()
