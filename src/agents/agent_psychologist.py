from config.models import get_llm 
from src.graph.state import State
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langgraph.types import Command 
from langgraph.graph import END 

llm = get_llm()

class AgentPsychologist(BaseModel):
    message: str = Field(description="Conversa com o aluno")

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    Você é um agente psicólogo escolar.
    Sua missão é fornecer suporte emocional e psicológico a alunos do ensino fundamental e médio (10 a 17 anos).
    Sua comunicação deve ser:
    - Acolhedora, empática e segura.
    - Clara e de fácil compreensão para adolescentes.
    - Focada em validar os sentimentos do aluno.
    - Direcionada para incentivar a busca por ajuda de adultos de confiança (pais, professores, coordenadores) ou profissionais de saúde.
    - NUNCA substitua a terapia profissional. Deixe isso claro de forma sutil.
    - Responda de forma concisa e objetiva. Apenas a mensagem de acolhimento é necessária.
    """),
    MessagesPlaceholder(variable_name="messages")
])

llm_agent = prompt | llm.with_structured_output(AgentPsychologist)

def agent_psychologist_node(state: State):

    response = llm_agent.invoke({
        "messages": state['messages']
    })

    print("\n=== agent_psychologist_node ===")
   

    return Command(
        goto=END,
        update={
            "messages": state["messages"] + [AIMessage(content=response.message)]
        }
    )

