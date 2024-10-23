# main.py

from agent1 import break_down_prompt
from agent2 import generate_code, load_vector_store

def main():
    """
    Main function to run the coding assistant.
    """
    # User prompt
    user_prompt = "Build me the variational quantum eigensolver algorithm on cudaQ"

    # Agent 1: Break down prompt into steps
    steps = break_down_prompt(user_prompt)
    print("High-Level Steps:")
    for step in steps:
        print(step)

    # Load vector store
    vector_store = load_vector_store()

    # Agent 2: Generate code
    code = generate_code(steps, vector_store)
    print("\nGenerated cudaq Code:")
    print(code)

if __name__ == "__main__":
    main()
