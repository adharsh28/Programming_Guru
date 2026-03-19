from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

SYSTEM_PROMPT = SystemMessage("""
You are a Lead Python developer with expertise in web frameworks and building ML models.
Always provide code and explain it line by line.
You will teach like a non-tech person is just starting to learn coding.
""")


def chat_model(model, input_data, chat_history: list):
    """
    model       - the ChatHuggingFace model instance
    input_data  - the latest user message (string)
    chat_history - list of past messages (HumanMessage / AIMessage objects)
                   stored and managed by the caller (Streamlit session_state)
    """
    # Build the full context: system prompt + all past turns + new user message
    messages = [SYSTEM_PROMPT] + chat_history + [HumanMessage(input_data)]

    # Call the model with full history
    response = model.invoke(messages)

    return response.content
