from langchain.memory import CombinedMemory, ConversationSummaryBufferMemory, ConversationEntityMemory
from langchain import OpenAI, LLMChain
from langchain.agents import ConversationalAgent, AgentExecutor, load_tools, initialize_agent
from modules.langchain.genericllm import WebUILLM

llm = WebUILLM()
#llm = OpenAI()
tools = load_tools([
    "python_repl",
    "requests",
    # "terminal",
    "llm-math",
    "wikipedia"
], llm=llm)

summary_window_memory = ConversationSummaryBufferMemory(
    memory_key="chat_history", llm=llm)
entity_memory = ConversationEntityMemory(
    chat_history_key="entity_history", llm=llm)

memory = CombinedMemory(memories=[
    summary_window_memory,
    # entity_memory
])

prefix = """Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:"""
suffix = """>>> Example:

New input: Hello!
Thought: Do I need to use a tool? No
AI: Hello!</end>

New input: Hi, what is 5!
Thought: Do I need to use a tool? Yes
Action: Calculator
Action Input: 5!
Observation: 120
Thought: Do I need to use a tool? No
AI: It's 120.</end>

>>> Scratchpad:

{agent_scratchpad}

>>> Begin! 

>>> Previous conversation history:

{chat_history}

New input: {input}
"""

prompt = ConversationalAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad", "chat_history"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
tool_names = [tool.name for tool in tools]
agent = ConversationalAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True)

#agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=memory)


def generate(message):
    response = agent_chain.run(f"{message}")
    if response.endswith('</end>'):
        response = response[:-6]
    return response, {}
