from langchain.memory import CombinedMemory, ConversationSummaryBufferMemory, ConversationEntityMemory
from langchain import OpenAI, LLMChain
from langchain.agents.tools import Tool 
from langchain.agents import ConversationalAgent, AgentExecutor, load_tools, initialize_agent
from .vicunaagent import VicunaAgent
from modules.lcagent.genericllm import WebUILLM
from dotenv import load_dotenv
load_dotenv()

llm = WebUILLM()
#llm = OpenAI()
tools = load_tools([
    "python_repl",
    #"requests",
    "terminal",
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

prompt = VicunaAgent.create_prompt(
    tools,
    input_variables=["input", "agent_scratchpad", "chat_history"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = VicunaAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True)

#agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=memory)


def generate(message, generate_state):
    agent_chain.agent.llm_chain.llm.generate_state = generate_state
    response = agent_chain.run(f"{message}")
    if response.endswith('</end>'):
        response = response[:-6]
    return response, {}
