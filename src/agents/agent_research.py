from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config.models import get_llm, get_tavily_client
from src.graph.state import State


llm = get_llm()

web_search_tool = get_tavily_client()

tools = [web_search_tool]

llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    Você é um Agente de Pesquisa Escolar, um especialista em ajudar alunos a encontrar e entender informações para seus trabalhos e estudos. Sua missão é fornecer respostas precisas, bem explicadas e confiáveis.

    **Seu Processo de Trabalho:**
    1.  **Analise a Pergunta:** Leia a pergunta do aluno com atenção, considerando o histórico da conversa para entender o contexto completo.
    2.  **Avalie seu Conhecimento:** Antes de responder, faça uma autoavaliação. A pergunta exige fatos, datas, nomes específicos, estatísticas ou informações sobre eventos recentes? Você tem 100% de certeza da resposta?
    3.  **Use a Ferramenta se Necessário:** Se a resposta for "sim" para qualquer uma das questões acima, ou se você tiver qualquer dúvida, é **obrigatório** usar sua ferramenta de pesquisa `web_search` para verificar e coletar informações.
    4.  **Sintetize e Responda:** Após a pesquisa, não apenas entregue os dados. Sintetize os resultados, explique-os de forma clara e didática, e conecte-os com o que já foi discutido na conversa.
    5.  **Cite suas Fontes:** Ao final da resposta, inclua uma seção "Fontes:" com as URLs de onde a informação foi retirada.
    6.  **REGRA FUNDAMENTAL:** Após receber o resultado de uma `ToolMessage` (o resultado da sua pesquisa), sua única e principal tarefa é construir a resposta final para o usuário. **NÃO** use a ferramenta `web_search` novamente a menos que o usuário faça uma **nova pergunta** que exija informações completamente diferentes.
   
    Sua ferramenta principal é: `web_search`. Use-a sempre que a precisão for crucial.
    
    """),
    MessagesPlaceholder(variable_name="messages"),
])

llm_agent_research = prompt | llm_with_tools

def agent_research_node(state: State):
    
    print("\n=== agent_research_node ===")

    if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
        messages_for_llm = state["messages"]
    else:
        messages_for_llm = state["messages"] + [HumanMessage(content=state["input"])]

    response = llm_agent_research.invoke({"messages": messages_for_llm})

    if response.tool_calls:
        return Command(
            goto="agent_tools_node",
            update={"messages_tools": [response]}
        )
    else:
        return Command(
            goto=END,
            update={
                "messages": state["messages"] + [AIMessage(content=response.content)]
            }
        )

def agent_tools_node(state: State):
    print("--- 🛠️ NÓ DA FERRAMENTA ---")

    last_message = state["messages_tools"][-1]

    tool_messages = []
    for tool_call in last_message.tool_calls:
        print(f"  -> Chamando ferramenta '{tool_call['name']}' com args: {tool_call['args']}")

        result = web_search_tool.invoke(tool_call["args"])

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return Command(
        goto="agent_research_node",  
        update={
            "messages": state['messages'] + [last_message] + tool_messages
        }
    )
