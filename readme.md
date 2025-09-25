# TUTOR-IA

Um tutor de inteligência artificial desenvolvido com LangChain, LangGraph e Tavily, capaz de responder a diferentes tipos de perguntas através de agentes especializados.

O sistema roteia a entrada do usuário para o agente mais adequado (matemática, psicologia, sentimentos, pesquisa, etc.), simulando um professor virtual inteligente.

---

## Funcionalidades

- Agente de sentimentos — detecta palavras sensivel do usuário. 
- Roteamento de agentes — o `agent_router` decide qual agente deve responder cada pergunta.  
- Agente de matemática — ajuda a resolver problemas e cálculos.  
- Agente de psicologia — atua como um conselheiro para reflexões pessoais.   
- Agente de pesquisa — busca informações externas com Tavily.  
- Agente de conversação — mantém diálogo natural com o usuário.  
- Agente de estudos — auxilia em questões educacionais.  
- Gerenciamento de estado — implementado com `graph/state.py`, mantém contexto entre interações.  

---

## Tecnologias utilizadas

- [LangChain](https://www.langchain.com/)  
- [LangGraph](https://python.langchain.com/docs/langgraph)  
- [Tavily](https://tavily.com/)  
- [Python 3.10+](https://www.python.org/)  
- [Virtualenv](https://docs.python.org/3/library/venv.html)  

---

## Estrutura do projeto

```
TUTOR-IA/
│── config/           # Configurações e modelos
│── src/
│   ├── agents/       # Agentes especializados
│   ├── graph/        # Fluxo e gerenciamento de estado
│── venv/             # Ambiente virtual (não versionado)
│── .env.example      # Exemplo de variáveis de ambiente
│── main.py           # Ponto de entrada da aplicação
```

---

## Como rodar o projeto

1. Clonar o repositório:
   ```bash
   git clone https://github.com/Hmachadoo/TUTOR-IA.git
   cd TUTOR-IA
   ```

2. Criar e ativar ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Instalar dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configurar variáveis de ambiente:
   - Copiar `.env.example` para `.env` e preencher com suas chaves de API (LangChain, Tavily, etc.).

5. Executar o projeto:
   ```bash
   python main.py
   ```

---

## Próximos passos

- Criar testes unitários com `pytest`.  
- Melhorar documentação dos agentes individuais.  
- Criar interface web simples para interação.  

---

## Autor

Projeto desenvolvido por **Henrique Machado** — estudante de tecnologia com foco em IA aplicada, Python e agentes inteligentes.
