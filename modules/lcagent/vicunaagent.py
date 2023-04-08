import json
import re
from typing import Any, List, Optional, Sequence, Tuple
from langchain import LLMChain, PromptTemplate
from langchain.agents import Agent
from langchain.tools.base import BaseTool
from langchain.llms import BaseLLM
from langchain.callbacks.base import BaseCallbackManager

PREFIX = """Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:"""

SUFFIX = """>>> Begin! Remember to use a tool to get factual information.

>>> Previous conversation history:

{chat_history}

>>> DATA:

{agent_scratchpad}

### Human: {input}
"""

FORMAT_INSTRUCTIONS = """All your responses must be in JSON format, with the following keys:

thought: The assistant thought process to choose the action
action: One of the actions. must be one of [{tool_names}, Respond]
query: The input to send to the action

If action is "respond", the query will be sent to the user as a message.

>>> Example tool usage:
### Human: hi
### Assistant:
{{{{
"thought": "The user greeted me. I should respond.",
"action": "Respond",
"query": "Hello, how may I assist you today?"
}}}}
### Human: how much is 10 * 10
### Assistant:
{{{{
"thought": "To perform this calculation, I will use the multiplication operation.",
"action": "Calculator",
"query": "10 + 10"
}}}}
### Assistant:
{{{{
"thought": "I now have the result.",
"action": "Respond",
"query": "10 + 10 equals to 20."
}}}}
# Result:
{{{{
"result": "Success"
}}}}
### Human: is this an arch-based linux system?
### Assistant:
{{{{
"thought": "I should run uname to get information on the OS.",
"action": "Terminal",
"query": "uname -r"
}}}}
# Result:
{{{{
"result": "6.2.8-arch1-1"
}}}}
### Assistant:
{{{{
"thought": "The system seems to be arch-based.",
"action": "Respond",
"query": "Yes, the system seems to be arch-based."
}}}}
# Result:
{{{{
"result": "Success"
}}}}

Do NOT send empty queries.
"""


class VicunaAgent(Agent):
    """An agent designed to hold a conversation in addition to using tools."""

    last_error: str = None

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "conversational-react-description-vicuna"

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "# Result:"

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "### Assistant:\n{"

    @property
    def _stop(self) -> List[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
            f"\n### Human:",
            f"\n\t### Human:",
            "}"
        ]

    def _fix_text(self, text: str) -> str:
        with open('fix-file.log', 'w') as f:
            f.write(text)

        return text + f"\n{{\n\"result\": \"{self.last_error}\"\n}}\n"

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            human_prefix: String to use before human output.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(
            tool_names=tool_names
        )
        template = "\n\n".join(
            [prefix, tool_strings, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)

    @property
    def finish_tool_name(self) -> str:
        """Name of the tool to use to finish the chain."""
        return "Respond"

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        json_start = llm_output.find('{')
        json_end = llm_output.rfind('}')
        response = None
        try:
            response = json.loads(llm_output[json_start:json_end + 1])
        except json.JSONDecodeError:
            return None

        action = response["action"]
        action_input = response["query"]

        return action.strip(), action_input

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLLM,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain, allowed_tools=tool_names, **kwargs
        )
