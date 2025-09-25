from typing_extensions import TypedDict

class State(TypedDict):
    input: str
    messages_tools: list
    messages_tools_math : list
    messages: list
    search_results: list
