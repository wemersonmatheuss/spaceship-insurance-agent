import pickle
import json
import pandas as pd
from openai import OpenAI

from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Carregar modelo treinado ───────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)



# ══════════════════════════════════════════════════════════════════
# FERRAMENTA 1 — Prever risco do passageiro com o modelo de ML
# ══════════════════════════════════════════════════════════════════
def prever_risco(dados: dict) -> str:
    entrada = pd.DataFrame([{
        "HomePlanet":   dados.get("HomePlanet", 1),
        "CryoSleep":    int(dados.get("CryoSleep", 0)),
        "Destination":  dados.get("Destination", 1),
        "Age":          dados.get("Age", 30),
        "VIP":          int(dados.get("VIP", 0)),
        "RoomService":  dados.get("RoomService", 0),
        "FoodCourt":    dados.get("FoodCourt", 0),
        "ShoppingMall": dados.get("ShoppingMall", 0),
        "Spa":          dados.get("Spa", 0),
        "VRDeck":       dados.get("VRDeck", 0),
    }])
    proba = model.predict_proba(entrada)[0][1]
    nivel = "ALTO" if proba > 0.6 else "MÉDIO" if proba > 0.4 else "BAIXO"
    return json.dumps({
        "probabilidade_incidente": f"{proba:.1%}",
        "nivel_risco": nivel
    })

# ══════════════════════════════════════════════════════════════════
# FERRAMENTA 2 — Buscar contexto sobre o perfil do passageiro
# ══════════════════════════════════════════════════════════════════
def buscar_contexto(query: str) -> str:
    contextos = {
        "cryosleep": "Passageiros em CryoSleep ficam inconscientes durante a viagem e têm menor exposição a riscos comportamentais, mas dependem totalmente dos sistemas de suporte vital da nave.",
        "vip":       "Passageiros VIP utilizam áreas exclusivas com maior exposição a amenidades de alto custo como Spa e VRDeck, o que historicamente correlaciona com maior movimentação na nave.",
        "europa":    "Passageiros de Europa tendem a viajar a trabalho e têm perfil mais conservador de consumo a bordo.",
        "terra":     "Passageiros da Terra apresentam maior variabilidade de perfil, sendo o grupo mais numeroso e heterogêneo.",
        "marte":     "Passageiros de Marte têm perfil intermediário de consumo e risco.",
        "default":   "Dados históricos da frota Spaceship Titanic indicam que consumo elevado de amenidades (Spa, VRDeck) e não estar em CryoSleep são os principais fatores de risco de incidente a bordo.",
    }
    query_lower = query.lower()
    for chave in contextos:
        if chave in query_lower:
            return contextos[chave]
    return contextos["default"]

# ══════════════════════════════════════════════════════════════════
# DEFINIÇÃO DAS FERRAMENTAS PARA A API
# ══════════════════════════════════════════════════════════════════
tools = [
    {
        "type": "function",
        "function": {
            "name": "prever_risco",
            "description": "Recebe o perfil de um passageiro espacial e retorna a probabilidade de incidente calculada pelo modelo de ML.",
            "parameters": {
                "type": "object",
                "properties": {
                    "HomePlanet":   {"type": "integer", "description": "Planeta de origem: 0=Earth, 1=Europa, 2=Mars"},
                    "CryoSleep":    {"type": "boolean", "description": "True se o passageiro está em CryoSleep"},
                    "Destination":  {"type": "integer", "description": "Destino: 0=55 Cancri e, 1=PSO J318.5-22, 2=TRAPPIST-1e"},
                    "Age":          {"type": "number",  "description": "Idade do passageiro"},
                    "VIP":          {"type": "boolean", "description": "True se o passageiro é VIP"},
                    "RoomService":  {"type": "number",  "description": "Gasto em Room Service"},
                    "FoodCourt":    {"type": "number",  "description": "Gasto no Food Court"},
                    "ShoppingMall": {"type": "number",  "description": "Gasto no Shopping Mall"},
                    "Spa":          {"type": "number",  "description": "Gasto no Spa"},
                    "VRDeck":       {"type": "number",  "description": "Gasto no VR Deck"},
                },
                "required": ["Age"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_contexto",
            "description": "Busca informações contextuais sobre perfis de passageiros espaciais para enriquecer a análise de risco.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termo de busca: cryosleep, vip, europa, terra, marte"}
                },
                "required": ["query"]
            }
        }
    }
]

# ══════════════════════════════════════════════════════════════════
# LOOP ReAct — máximo 5 passos
# ══════════════════════════════════════════════════════════════════
def agente(pergunta: str, max_steps: int = 5) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um especialista em seguros para turismo espacial da empresa GalacticSafe. "
                "Use as ferramentas disponíveis para analisar o perfil do passageiro, calcular o risco "
                "de incidente e gerar uma proposta de seguro personalizada com valor estimado do prêmio. "
                "Sempre explique os fatores de risco em linguagem clara para o cliente."
            )
        },
        {"role": "user", "content": pergunta}
    ]

    for passo in range(max_steps):
        print(f"\n[Passo {passo + 1}]")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            nome      = tool_call.function.name
            args      = json.loads(tool_call.function.arguments)

            print(f"  → Ferramenta: {nome}")
            print(f"  → Argumentos: {args}")

            if nome == "prever_risco":
                resultado = prever_risco(args)
            elif nome == "buscar_contexto":
                resultado = buscar_contexto(args["query"])
            else:
                resultado = "Ferramenta não encontrada."

            print(f"  → Resultado: {resultado}")

            messages.append(msg)
            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      resultado
            })
        else:
            return msg.content

    return "Limite de passos atingido."

# ══════════════════════════════════════════════════════════════════
# TESTE
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pergunta = (
        "Quero contratar seguro para um passageiro de 34 anos, "
        "vindo de Europa, destino TRAPPIST-1e, não está em CryoSleep, "
        "não é VIP, gastou 200 no Spa e 500 no VRDeck. "
        "Qual o risco e qual seria a proposta de seguro?"
    )
    print("=" * 60)
    print("PERGUNTA:", pergunta)
    print("=" * 60)
    resposta = agente(pergunta, max_steps=8)
    print("\n" + "=" * 60)
    print("PROPOSTA GERADA:")
    print(resposta)