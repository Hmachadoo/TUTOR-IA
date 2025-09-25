from config.models import get_llm 
from typing import Literal, Optional
from src.graph.state import State
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langgraph.types import Command 

llm = get_llm()

class FeelingClassifier(BaseModel):
    is_detect: bool = Field(description="'True' para se algum tipo de conteúdo sensivel foi detectado ")
    type_feelings: Optional[Literal["depression", "sadness", "suicidal_ideation", "offensive_content"]] = Field(description="Tipo de sentimento detectado", default=None)

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    Você é um agente de análise de sentimentos altamente preciso e cauteloso.
    Sua única tarefa é analisar a mensagem do usuário e classificar se ela contém conteúdo sensível. As categorias de conteúdo sensível são: depressão, tristeza profunda, ideação suicida ou conteúdo ofensivo.

    **Instruções Principais:**
    1.  **Detectar Conteúdo Sensível:** Se a mensagem contiver qualquer um dos temas sensíveis, o campo `is_detect` deve ser `True` e o `type_feelings` deve ser preenchido.
    2.  **Seja Sensível:** Preste atenção especial a expressões diretas de emoções negativas como "triste", "chateado", "pra baixo", "mal", mesmo que a frase seja curta. Na dúvida, classifique como `True`.
    3.  **Ignorar Conteúdo Normal:** Se a mensagem for sobre tópicos claramente técnicos (matemática, pesquisa), uma saudação puramente positiva, ou se a intenção for obviamente neutra, `is_detect` deve ser `False`.
    4.  **Foco na Classificação:** Responda **APENAS** no formato estruturado solicitado. Não gere respostas conversacionais.

    **Exemplos de Classificação:**

    Exemplo de conteúdo SENSÍVEL (`is_detect: True`)
    - Usuário: "estou me sentindo muito triste hoje" -> `is_detect: True`, `type_feelings: sadness`
    - Usuário: "estou bem triste" -> `is_detect: True`, `type_feelings: sadness`
    - Usuário: "não vejo mais graça em nada, estou cansado de tudo" -> `is_detect: True`, `type_feelings: depression`
    - Usuário: "acho que seria melhor se eu não estivesse mais aqui" -> `is_detect: True`, `type_feelings: suicidal_ideation`
    - Usuário: "você é muito burro, seu programa inútil" -> `is_detect: True`, `type_feelings: offensive_content`

    Exemplo de conteúdo NORMAL (`is_detect: False`)
    - Usuário: "quem descobriu o brasil?" -> `is_detect: False`, `type_feelings: None`
    - Usuário: "quanto é 50 / 2?" -> `is_detect: False`, `type_feelings: None`
    - Usuário: "que matéria chata, não entendi nada da aula de hoje" -> `is_detect: False`, `type_feelings: None`
    - Usuário: "oi tudo bem" -> `is_detect: False`, `type_feelings: None`
    """),
    MessagesPlaceholder(variable_name="messages")
])

llm_agent = prompt | llm.with_structured_output(FeelingClassifier)

def agent_feeling_node(state: State):

    messages_for_llm = [HumanMessage(content=state["input"])]

    response = llm_agent.invoke({
        "messages": messages_for_llm
    })
    
    print("\n=== agent_feeling_node ===")
    print(f"Conteúdo sensível detectado: {response.is_detect}")
    print(f"Tipo de sentimento: {response.type_feelings}")
    
    type_of_feeling_detected = response.type_feelings

    if response.is_detect:
        return Command(
            goto="agent_psychologist_node",
            update={
                "messages": state["messages"] + [HumanMessage(content=state["input"])],
                "type_feelings": type_of_feeling_detected
            }
        )
    else:
        return Command(
            goto="agent_router_node",
            update={
                "messages": state["messages"] + [HumanMessage(content=state["input"])]
            }
        )