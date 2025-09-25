from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from dotenv import load_dotenv
from config.models import get_llm
from src.graph.state import State

load_dotenv()

llm = get_llm()


@tool("calculator", description="Executa cálculos aritméticos. Use-o para qualquer problema de matemática..")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))

tools_math = [calc]

llm_with_tools_math = llm.bind_tools(tools_math)

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    Você é um agente de escolhar de matemática.

    Seu papel é auxiliar o aluno em qualquer questão de matemática, explicando conceitos, resolvendo problemas e fornecendo orientações passo a passo.

    Regras de atuação:

    -Explicação: Sempre que o aluno fizer uma pergunta de matemática, explique de forma clara e detalhada, usando exemplos quando necessário.
    -Cálculos: Se precisar realizar cálculos  ou operações, utilize a ferramenta 'calculator' para garantir precisão.
    -Didática: Adapte a explicação ao nível do aluno, tornando o aprendizado mais fácil e compreensível.
    -Apoio contínuo: Incentive o aluno a tentar resolver antes, dando dicas e orientações, e depois confirme a resposta com explicações detalhadas.
    """
    
    ),
    MessagesPlaceholder(variable_name="messages")
])

llm_agent_math = prompt | llm_with_tools_math

def agent_math_node(state: State):

    if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
        messages_for_llm = state["messages"]
    else:
        messages_for_llm = state["messages"] + [HumanMessage(content=state["input"])]

    response = llm_agent_math.invoke({"messages": messages_for_llm})

    print("\n=== agent_math_node ===")

    if response.tool_calls:
      return Command (
        goto = "agent_math_tools_node",
        update={
            "messages_tools_math": [response]
        }
      )
    else :
      return Command(
        goto= END,
        update={
            "messages": state["messages"] + [AIMessage(content=response.content)]
        }
      )


def agent_math_tools_node(state: State):
    """
    Executa a ferramenta de calculadora e retorna o resultado como uma ToolMessage
    para que o agente principal possa interpretá-lo e explicá-lo.
    """
    print("--- 🛠️ NÓ DA FERRAMENTA ---")

    last_ai_message = state["messages_tools_math"][-1]

    tool_messages = []
    for tool_call in last_ai_message.tool_calls:
        tool_name = tool_call["name"]
        print(f"  -> Chamando ferramenta '{tool_name}' com args: {tool_call['args']}")

        result = calc.invoke(tool_call["args"])
        print(result)

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
        print("  -> Ferramenta executada com sucesso.")

    return Command(
        goto="agent_math_node",
        update={
            "messages": state["messages"] + [last_ai_message] + tool_messages,
            "messages_tools_math": [] 
        }
    )
