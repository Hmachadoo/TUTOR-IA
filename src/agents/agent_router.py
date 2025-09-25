from typing import Literal, Optional
from config.models import get_llm 
from src.graph.state import State
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command 

llm = get_llm()

class AgentRouter(BaseModel):
    route : Optional[Literal["math", "conversation", "researcher", "study_mode"]] = Field (description= "Indentificar o assunto do aluno ")

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     Você é um agente roteador. 
     Sua tarefa é analisar a última mensagem do aluno e identificar o assunto principal. 
     Considere o contexto da mensagem e escolha apenas **uma** das seguintes categorias:

     - "math": topicos relacioando a matemática ou cálculos.
     - "conversation": uma saudação, assuntos de bate-papo ou conversas gerais.
     - "researcher": temas de pesquisa , história, duvidas sobre fatos historicos.
     - "study_mode": gerar questões ,tópicos de estudo, revisão ou aprendizado de conteúdo.

     **Instruções importantes:**
     1. Retorne apenas uma das opções acima, sem explicações adicionais.
     2. Seja preciso na escolha, considerando o foco principal da mensagem.

    **SEMPRE** reflita até encontrar um tema para rotear, **NUNCA** retorne nulo 
     """),
    MessagesPlaceholder(variable_name="messages")
])

llm_agent_router = prompt | llm.with_structured_output(AgentRouter)

def agent_router_node(state: State):

    response = llm_agent_router.invoke({
        "messages": state["messages"] + [HumanMessage(content=state["input"])]
    })

    print("\n=== agent_router_node ===")
    print("Assunto identificado pelo roteador:", response.route)

    route = response.route

    if route == "math":
        goto = "agent_math_node"
    elif route == "researcher":
        goto = "agent_research_node"
    elif route == "study_mode":
        goto = "agent_study_mode_node"
    else:
        goto = "agent_conversation_node"

    return Command(
        goto=goto,
        update={
            "messages": state["messages"] + [AIMessage(content=f"Assunto: {route}")]
        }
    )
