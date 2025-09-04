from langchain_google_genai import GoogleGenerativeAI

def generate_response(query, retrieved_chunks, help_type):
    """Use Gemini model to generate a response based on content type."""
    llm = GoogleGenerativeAI(model="gemini-2.0-flash")
    
    if help_type == "Topic Explanations":
        prompt = (
            f"User Query: {query}\n"
            f"Retrieved Content: {' '.join(retrieved_chunks)}\n"
            "Generate a detailed and structured explanation suitable for students."
        )
    elif help_type == "Exam Questions":
        prompt = (
            f"Content: {' '.join(retrieved_chunks)}\n"
            f"Create 5 exam questions with answers based on this content."
        )
    elif help_type == "Assignment":
        prompt = (
            f"Based on: {' '.join(retrieved_chunks)}\n"
            "Create a student assignment with tasks/questions and instructions."
        )
    elif help_type == "Interview Questions":
        prompt = (
            f"Topic Context: {' '.join(retrieved_chunks)}\n"
            "Generate technical interview questions (with answers) relevant to this topic."
        )
    else:
        prompt = f"Query: {query}\nContent: {' '.join(retrieved_chunks)}"

    return llm.invoke(prompt)

def is_educational_with_llm(query):
    """Use LLM to classify if the query is educational."""
    llm = GoogleGenerativeAI(model="gemini-2.0-flash")
    prompt = (
        f"Is the following user query related to education or learning? "
        f"Respond only with 'yes' or 'no'.\n\nQuery: {query}"
    )
    response = llm.invoke(prompt)
    return "yes" in response.lower()
