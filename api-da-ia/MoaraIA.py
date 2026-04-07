import os
from datetime import datetime
from dotenv import load_dotenv

# Componentes (API para IA)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

class MoaraIA:
    """
    Classe para gerenciar a inteligência artificial Moara, 
    encapsulando LLMs, ferramentas e memória de curto prazo.
    """
    
    def __init__(self, tools=None):
        load_dotenv()
        self.tools = tools or []
        self.memory = MemorySaver()
        self.system_prompt = self._load_system_prompt()
        self.llm = self._setup_llm()
        self.agent = self._setup_agent()

    def _load_system_prompt(self) -> str:
        """Carrega o prompt do sistema do arquivo local."""
        try:
            with open("systemprompt.txt", "r", encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print("Aviso: 'systemprompt.txt' não encontrado. Usando prompt padrão.")
            return "Você é a Moara, uma assistente prestativa."

    def _setup_llm(self):
        """Configura o modelo principal com fallback para outro provedor."""
        set_temperature = 0.4
        set_top_p = 0.9

        # Modelo Principal: Gemini
        llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # Versão estável atualizada
            temperature=set_temperature,
            top_p=set_top_p,
            google_api_key=os.getenv("GEMINI_API")
        )

        # Modelo de Fallback: Groq (Llama)
        llm_groq = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=set_temperature,
            top_p=set_top_p,
            groq_api_key=os.getenv("GROQ_API")
        )

        return llm_gemini.with_fallbacks([llm_groq])

    def _setup_agent(self):
        """Inicializa o agente com suporte a ferramentas e memória."""
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=self.memory
        )

    def out_game_response(self, pergunta: str, thread_id: str = "default_user") -> str:
        """
        Processa uma pergunta e retorna a resposta final do agente.
        
        Args:
            pergunta: O texto enviado pelo usuário.
            thread_id: Identificador da conversa para manter o contexto.
        """

        pergunta = "(out_game) -> "+pergunta

        try:
            # Invoca o agente com o contexto temporal
            input_data = {
                "messages": [{
                    "role": "user",
                    "content": f"{pergunta}\n(AGORA: {datetime.now()})"
                }]
            }
            
            config = {"configurable": {"thread_id": thread_id}}
            result = self.agent.invoke(input_data, config=config)
            
            # Extração robusta do conteúdo da última mensagem
            last_message = result['messages'][-1]
            content = last_message.content

            # Tratamento para casos onde o conteúdo vem como lista de blocos (comum em multimodais)
            if isinstance(content, list):
                text_parts = [item.get('text', '') for item in content if isinstance(item, dict)]
                return "".join(text_parts).strip()
            
            return content.strip()

        except Exception as e:
            return f"Erro ao processar resposta: {str(e)}"
        
    def in_game_response(self, pergunta: str, thread_id: str = "default_user") -> str:
        """
        Args:
            pergunta: O texto enviado pelo usuário.
            thread_id: Identificador da conversa para manter o contexto.
        """
        
        pergunta = "(in_game) -> "+pergunta

        try:
            # Invoca o agente com o contexto temporal
            input_data = {
                "messages": [{
                    "role": "user",
                    "content": f"{pergunta}\n(AGORA: {datetime.now()})"
                }]
            }
            
            config = {"configurable": {"thread_id": thread_id}}
            result = self.agent.invoke(input_data, config=config)
            
            # Extração robusta do conteúdo da última mensagem
            last_message = result['messages'][-1]
            content = last_message.content

            # Tratamento para casos onde o conteúdo vem como lista de blocos (comum em multimodais)
            if isinstance(content, list):
                text_parts = [item.get('text', '') for item in content if isinstance(item, dict)]
                return "".join(text_parts).strip()
            
            return content.strip()

        except Exception as e:
            return f"Erro ao processar resposta: {str(e)}"
        
    