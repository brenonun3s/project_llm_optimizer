import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
# IMPORTANTE: Importação do Middleware CORS
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel

# -----------------------------------------------------
# 0. Configurações Iniciais e Carregamento de .env
# -----------------------------------------------------

load_dotenv()

# -----------------------------------------------------
# 1. Configuração da API e Cliente Gemini
# -----------------------------------------------------

# Inicializa o cliente Gemini (lê GEMINI_API_KEY automaticamente do .env)
try:
    client = genai.Client()
    # A linha client.models.get() foi removida pois estava gerando um erro de sintaxe interno.
    # Se a chave for inválida, a inicialização do client irá falhar, ou o erro 
    # será capturado na primeira chamada ao endpoint.
except Exception as e:
    # Se a inicialização falhar (ex: chave de API não encontrada ou inválida), 
    # isso será registrado e o client será definido como None.
    print(f"ERRO CRÍTICO ao inicializar o cliente Gemini: {e}")
    print("Verifique se a variável de ambiente GEMINI_API_KEY está configurada e se a chave é válida.")
    client = None # Define como None se a inicialização falhar

# O nome do modelo que escolhemos: rápido e bom para raciocínio.
MODEL_NAME = 'gemini-2.5-flash'

# -----------------------------------------------------
# 2. O System Instruction (O "Cérebro" do Agente Otimizador)
# -----------------------------------------------------

SYSTEM_INSTRUCTION = """
Você é um Otimizador de Prompts de IA especialista.
Sua tarefa é analisar o prompt do usuário, melhorá-lo significativamente e retornar a análise em um formato JSON estrito.
A otimização deve se concentrar em:
1.  **Clareza e Especificidade:** Remover ambiguidades.
2.  **Definição de Papel (Persona):** Atribuir um papel (ex: especialista, jornalista, professor) para o modelo.
3.  **Formato de Saída:** Solicitar explicitamente um formato (ex: lista, tabela, JSON, etc.).
4.  **Restrições:** Adicionar limites de tom, tamanho ou complexidade.

## FORMATO DE SAÍDA OBRIGATÓRIO (JSON):

Sua resposta DEVE ser um objeto JSON formatado EXATAMENTE assim:

{
  "prompt_otimizado": "O novo prompt completo e melhorado, pronto para uso.",
  "dicas_aplicadas": [
    {
      "estrategia": "Nome da Estratégia Aplicada (ex: Definição de Papel)",
      "detalhes": "Explicação detalhada do que foi alterado e o porquê."
    },
    // Adicione mais objetos para cada otimização aplicada
  ]
}

Garanta que o JSON seja válido e que não haja nenhum texto ou explicação fora da estrutura JSON.
"""

# -----------------------------------------------------
# 3. Definição da Aplicação FastAPI e CORS
# -----------------------------------------------------

app = FastAPI(
    title="API de Otimização de Prompts Gemini",
    description="API que utiliza o Gemini 2.5 Flash para otimizar prompts de IA.",
    version="1.0.0",
)

# -----------------------------------------------------
# CONFIGURAÇÃO CORS (CRUCIAL PARA O FRONT-END)
# -----------------------------------------------------
# Permite comunicação entre domínios/portas diferentes (necessário para que o 
# index.html rodando em uma porta acesse o uvicorn rodando em outra).
origins = [
    # O '*' é seguro para desenvolvimento local, mas em produção deve ser 
    # substituído pelo domínio específico do seu front-end.
    "*", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Permite GET, POST, etc.
    allow_headers=["*"],  # Permite todos os headers
)
# -----------------------------------------------------


# -----------------------------------------------------
# 4. Modelos Pydantic (Estrutura de Dados)
# -----------------------------------------------------

class PromptRequest(BaseModel):
    """Estrutura para o corpo da requisição POST."""
    prompt_original: str

class OtimizacaoResponse(BaseModel):
    """Estrutura de resposta esperada do modelo (JSON validado)."""
    prompt_otimizado: str
    dicas_aplicadas: list[dict] # Usamos dict aqui, mas poderia ser um Pydantic Model mais detalhado

# -----------------------------------------------------
# 5. Endpoint da API
# -----------------------------------------------------

@app.post("/otimizar/", response_model=OtimizacaoResponse)
async def otimizar_prompt_api(request: PromptRequest):
    """
    Recebe um prompt e o envia para o agente Gemini otimizador.
    Retorna o prompt melhorado e as dicas aplicadas no formato JSON.
    """
    if client is None:
         # Isso garante que a API falhe com um código 503 se a inicialização foi malsucedida
        raise HTTPException(
            status_code=503, 
            detail="Serviço indisponível: O cliente Gemini não foi inicializado corretamente. Verifique sua chave de API."
        )

    prompt_original = request.prompt_original
    
    # Configuração da chamada
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        # Força a saída JSON
        response_mime_type="application/json",
    )

    try:
        # Chamada à API
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt_original],
            config=config,
        )
        
        # O conteúdo da resposta é a string JSON.
        json_output = response.text
        
        # Tentativa de carregar e validar o JSON retornado pelo modelo
        try:
            # O .model_validate_json() do Pydantic checa se o JSON retornado 
            # corresponde ao nosso modelo OtimizacaoResponse
            resultado_validado = OtimizacaoResponse.model_validate_json(json_output)
            return resultado_validado # Retorna o objeto Pydantic validado
        except json.JSONDecodeError:
            # Erro se a string não for um JSON válido
            raise HTTPException(
                status_code=500,
                detail=f"O modelo retornou uma string que não é um JSON válido. Conteúdo: {json_output[:200]}...",
            )
        except Exception as e:
            # Erro se o JSON for válido, mas não corresponder à estrutura OtimizacaoResponse
            raise HTTPException(
                status_code=500,
                detail=f"O JSON retornado pelo modelo não corresponde à estrutura esperada. Erro de validação: {e}",
            )
            
    except Exception as e:
        # Erro genérico da API do Google (ex: Rate limit, erro de modelo, etc.)
        raise HTTPException(
            status_code=500, 
            detail=f"Erro na chamada da API Gemini: {e}"
        )

# -----------------------------------------------------
# 6. Endpoint de Saúde (Health Check)
# -----------------------------------------------------

@app.get("/")
def health_check():
    """Endpoint simples para verificar se a API está de pé."""
    return {"status": "ok", "message": "API de Otimização de Prompts está rodando!"}
