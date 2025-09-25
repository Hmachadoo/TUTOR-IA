from langgraph.graph import StateGraph, END
from src.graph.state import State
from src.agents.agent_feeling import agent_feeling_node
from src.agents.agent_psychologist import agent_psychologist_node
from src.agents.agent_router import agent_router_node
from src.agents.agent_conversation import agent_conversation_node
from src.agents.agent_math import agent_math_node 
from src.agents.agent_math import agent_math_node, agent_math_tools_node
from src.agents.agent_research import agent_research_node,agent_tools_node
from src.agents.agent_study import agent_study_mode_node

graph_builder = StateGraph(State)

graph_builder.add_node("agent_feeling_node", agent_feeling_node)
graph_builder.add_node("agent_psychologist_node", agent_psychologist_node)
graph_builder.add_node("agent_router_node", agent_router_node)
graph_builder.add_node("agent_conversation_node", agent_conversation_node)
graph_builder.add_node("agent_research_node", agent_research_node)
graph_builder.add_node("agent_tools_node", agent_tools_node)
graph_builder.add_node("agent_math_node", agent_math_node)
graph_builder.add_node("agent_math_tools_node", agent_math_tools_node)
graph_builder.add_node("agent_study_mode_node", agent_study_mode_node)


graph_builder.set_entry_point("agent_feeling_node")

app = graph_builder.compile()