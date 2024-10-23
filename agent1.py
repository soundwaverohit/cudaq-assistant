from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

def break_down_prompt(prompt):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

    breakdown_prompt = PromptTemplate(
        input_variables=['prompt'],
        template="""
You are an expert quantum computing assistant.
Break down the following prompt into a numbered list of high-level steps required to implement it:

Prompt: "{prompt}"

Steps:
"""
    )
    formatted_prompt = breakdown_prompt.format(prompt=prompt)
    response = llm.predict(formatted_prompt)
    steps = response.strip().split('\n')

    steps = [step.strip() for step in steps if step.strip()]
    return steps

if __name__ == "__main__":
    user_prompt = "Build me the variational quantum eigensolver algorithm on cudaQ"
    steps = break_down_prompt(user_prompt)
    print("Steps:")
    for step in steps:
        print(step)
