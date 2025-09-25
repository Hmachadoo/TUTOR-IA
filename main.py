from src.graph.builder import app
from langchain_core.messages import AIMessage

def main():
    state = {"messages": []}

    print(" Olá, sou o chatbot da escola. Digite 'sair' para encerrar.\n")

    while True:
        user_input = input("Você: ")

        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Até logo!")
            break

        state["input"] = user_input

        state = app.invoke(state)

        last_msg = state["messages"][-1] if state["messages"] else None
        if isinstance(last_msg, AIMessage):
            print(f"Agente: {last_msg.content}")


if __name__ == "__main__":
    main()