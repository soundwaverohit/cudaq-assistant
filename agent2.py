# agent2.py

from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

def load_vector_store():
    """
    Load the pre-built FAISS vector store.

    Returns:
        FAISS: The loaded vector store.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        '/Users/rohitganti/Desktop/cudaq-assistant/aa1/vector_store',
        embeddings,
        allow_dangerous_deserialization=True  # Use this if necessary
    )
    return vector_store

def generate_code(steps, vector_store):
    """
    Generate cudaq code for each step using in-context learning.

    Args:
        steps (list): List of steps from Agent 1.
        vector_store (FAISS): The vector store for similarity search.

    Returns:
        str: The generated cudaq code.
    """
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    code = ""

    for step in steps:
        # Perform similarity search
        docs = vector_store.similarity_search(step, k=3)

        # Prepare context from similar instructions
        context = ""
        for doc in docs:
            context += f"Instruction: {doc.page_content}\nResponse: {doc.metadata['response']}\n\n"

        # Create prompt for code generation
        code_prompt = PromptTemplate(
            input_variables=['context', 'step'],
            template="""
You are a CUDA Quantum (cudaq) coding assistant.
Use the following relevant instructions and code snippets as context to help generate the code for the given step.

Context:
{context}

Step:
{step}

Write the code in cudaq to perform this step, utilizing the context as appropriate.
"""
        )
        formatted_prompt = code_prompt.format(context=context, step=step)
        response = llm.predict(formatted_prompt)
        code += f"# {step}\n{response}\n\n"

    return code

if __name__ == "__main__":
    from agent1 import break_down_prompt

    user_prompt = "Build me the variational quantum eigensolver algorithm on cudaQ"
    steps = break_down_prompt(user_prompt)
    vector_store = load_vector_store()
    code = generate_code(steps, vector_store)
    print("Generated Code:")
    print(code)
