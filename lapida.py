import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------
# 1. Configuração da API
# -----------------------------------------------------

# A chave de API deve ser configurada como uma variável de ambiente (ex: export GEMINI_API_KEY="SUA_CHAVE").
# O cliente genai a lerá automaticamente.
try:
    client = genai.Client()
except Exception as e:
    print(f"Erro ao inicializar o cliente: {e}")
    print("Certifique-se de que a variável de ambiente GEMINI_API_KEY está configurada corretamente.")
    exit()

# O nome do modelo que escolhemos: rápido e bom para raciocínio.
MODEL_NAME = 'gemini-2.5-flash'

# -----------------------------------------------------
# 2. O System Instruction (O "Cérebro" do Agente Otimizador)
# -----------------------------------------------------

SYSTEM_INSTRUCTION = """
Você é um Otimizador de Prompts de IA especialista.
Sua tarefa é analisar o prompt do usuário, melhorá-lo significativamente e retornar a análise em um formato JSON estrito.
A otimização deve se concentrar em:
1.  **Clareza e Especificidade:** Remover ambiguidades.
2.  **Definição de Papel (Persona):** Atribuir um papel (ex: especialista, jornalista, professor) para o modelo.
3.  **Formato de Saída:** Solicitar explicitamente um formato (ex: lista, tabela, JSON, etc.).
4.  **Restrições:** Adicionar limites de tom, tamanho ou complexidade.

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
# 3. Função Principal
# -----------------------------------------------------

def otimizar_prompt(prompt_original: str):
    """
    Envia o prompt original para o Agente Gemini e retorna a saída otimizada.
    """
    print(f"--- PROMPT ORIGINAL ---\n{prompt_original}\n")

    # Configuração da chamada
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        # Força a saída JSON para garantir a estrutura do nosso agente
        response_mime_type="application/json",
    )

    # Chamada à API
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt_original],
            config=config,
        )
        
        # O conteúdo da resposta é a string JSON
        json_output = response.text
        
        return json_output

    except Exception as e:
        return f"Erro na chamada da API: {e}"

# -----------------------------------------------------
# 4. Execução
# -----------------------------------------------------

if __name__ == "__main__":
    # Prompt do usuário para teste
    prompt_teste = input("Digite o prompt a ser melhorado:\n")

    resultado_json = otimizar_prompt(prompt_teste)
    
    print("--- RESULTADO JSON RECEBIDO ---")
    print(resultado_json)
    
    # OPCIONAL: Para facilitar a visualização no Python, podemos carregar o JSON
    try:
        import json
        resultado = json.loads(resultado_json)
        
        print("\n" + "="*50)
        print("AGENT FÁCIL DE LER (PARSED):")
        print("="*50)
        print(f"PROMPT OTIMIZADO:\n{resultado['prompt_otimizado']}")
        
        print("\n--- DICAS APLICADAS ---")
        for dica in resultado['dicas_aplicadas']:
            print(f"- **{dica['estrategia']}**: {dica['detalhes']}")
            
    except json.JSONDecodeError:
        print("\nERRO: O modelo não retornou um JSON válido. Verifique o System Instruction.")
    except Exception as e:
        print(f"\nOcorreu um erro ao processar o resultado: {e}")