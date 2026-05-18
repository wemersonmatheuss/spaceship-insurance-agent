import streamlit as st
import pickle
import json
import pandas as pd
from openai import OpenAI

from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Configuração da página ─────────────────────────────────────────
st.set_page_config(
    page_title="GalacticSafe — Seguros Espaciais",
    page_icon="🚀",
    layout="centered"
)

# ── Carregar modelo ────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    import os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import pickle

    # Treina o modelo na hora se não tiver o pkl
    df = pd.read_csv("data/train.csv")
    features = ["HomePlanet", "CryoSleep", "Destination", "Age",
                "VIP", "RoomService", "FoodCourt", "ShoppingMall",
                "Spa", "VRDeck"]
    df = df[features + ["Transported"]].dropna()
    le = LabelEncoder()
    for col in ["HomePlanet", "Destination"]:
        df[col] = le.fit_transform(df[col])
    df["CryoSleep"]   = df["CryoSleep"].astype(int)
    df["VIP"]         = df["VIP"].astype(int)
    df["Transported"] = df["Transported"].astype(int)
    X = df[features]
    y = df["Transported"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


# ══════════════════════════════════════════════════════════════════
# FERRAMENTAS
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
        "nivel_risco": nivel,
        "proba_raw": round(proba, 4)
    })

def buscar_contexto(query: str) -> str:
    contextos = {
        "cryosleep": "Passageiros em CryoSleep ficam inconscientes durante a viagem e têm menor exposição a riscos comportamentais, mas dependem totalmente dos sistemas de suporte vital da nave.",
        "vip":       "Passageiros VIP utilizam áreas exclusivas com maior exposição a amenidades de alto custo como Spa e VRDeck, o que historicamente correlaciona com maior movimentação na nave.",
        "europa":    "Passageiros de Europa tendem a viajar a trabalho e têm perfil mais conservador de consumo a bordo.",
        "terra":     "Passageiros da Terra apresentam maior variabilidade de perfil, sendo o grupo mais numeroso e heterogêneo.",
        "marte":     "Passageiros de Marte têm perfil intermediário de consumo e risco.",
        "default":   "Dados históricos da frota Spaceship Titanic indicam que consumo elevado de amenidades (Spa, VRDeck) e não estar em CryoSleep são os principais fatores de risco de incidente a bordo.",
    }
    for chave in contextos:
        if chave in query.lower():
            return contextos[chave]
    return contextos["default"]

tools = [
    {
        "type": "function",
        "function": {
            "name": "prever_risco",
            "description": "Recebe o perfil de um passageiro espacial e retorna a probabilidade de incidente calculada pelo modelo de ML.",
            "parameters": {
                "type": "object",
                "properties": {
                    "HomePlanet":   {"type": "integer", "description": "0=Earth, 1=Europa, 2=Mars"},
                    "CryoSleep":    {"type": "boolean"},
                    "Destination":  {"type": "integer", "description": "0=55 Cancri e, 1=PSO J318.5-22, 2=TRAPPIST-1e"},
                    "Age":          {"type": "number"},
                    "VIP":          {"type": "boolean"},
                    "RoomService":  {"type": "number"},
                    "FoodCourt":    {"type": "number"},
                    "ShoppingMall": {"type": "number"},
                    "Spa":          {"type": "number"},
                    "VRDeck":       {"type": "number"},
                },
                "required": ["Age"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_contexto",
            "description": "Busca informações contextuais sobre perfis de passageiros espaciais.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

def agente(pergunta: str, max_steps: int = 8) -> tuple[str, list]:
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
    log_passos = []
    for passo in range(max_steps):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            nome  = tool_call.function.name
            args  = json.loads(tool_call.function.arguments)
            if nome == "prever_risco":
                resultado = prever_risco(args)
            elif nome == "buscar_contexto":
                resultado = buscar_contexto(args["query"])
            else:
                resultado = "Ferramenta não encontrada."
            log_passos.append({"passo": passo + 1, "ferramenta": nome, "resultado": resultado})
            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": resultado
            })
        else:
            return msg.content, log_passos
    return "Limite de passos atingido.", log_passos

# ══════════════════════════════════════════════════════════════════
# INTERFACE
# ══════════════════════════════════════════════════════════════════
st.title("🚀 GalacticSafe")
st.subheader("Seguro Espacial Personalizado com IA")
st.markdown("Preencha o perfil do passageiro e o agente de IA irá calcular o risco e gerar uma proposta de seguro.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 Perfil do Passageiro")
    idade       = st.number_input("Idade", min_value=1, max_value=120, value=34)
    planeta     = st.selectbox("Planeta de Origem", ["Europa", "Terra", "Marte"])
    destino     = st.selectbox("Destino", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
    cryosleep   = st.toggle("Em CryoSleep?")
    vip         = st.toggle("Passageiro VIP?")

with col2:
    st.markdown("#### 💳 Gastos nas Amenidades (GalacticCredits)")
    room_service  = st.number_input("Room Service",   min_value=0, value=0)
    food_court    = st.number_input("Food Court",     min_value=0, value=0)
    shopping_mall = st.number_input("Shopping Mall",  min_value=0, value=0)
    spa           = st.number_input("Spa",            min_value=0, value=200)
    vrdeck        = st.number_input("VR Deck",        min_value=0, value=500)

st.divider()

planeta_map  = {"Europa": 1, "Terra": 0, "Marte": 2}
destino_map  = {"TRAPPIST-1e": 2, "55 Cancri e": 0, "PSO J318.5-22": 1}

if st.button("🔍 Analisar Risco e Gerar Proposta", use_container_width=True):
    pergunta = (
        f"Quero contratar seguro para um passageiro de {idade} anos, "
        f"vindo de {planeta}, destino {destino}, "
        f"{'está' if cryosleep else 'não está'} em CryoSleep, "
        f"{'é' if vip else 'não é'} VIP, "
        f"gastou {room_service} em Room Service, {food_court} no Food Court, "
        f"{shopping_mall} no Shopping Mall, {spa} no Spa e {vrdeck} no VRDeck. "
        f"Qual o risco e qual seria a proposta de seguro?"
    )

    with st.spinner("Agente analisando o perfil..."):
        proposta, log = agente(pergunta)

    # ── Resultado do modelo ────────────────────────────────────────
    risco_raw = None
    for entry in log:
        if entry["ferramenta"] == "prever_risco":
            dados_risco = json.loads(entry["resultado"])
            risco_raw   = dados_risco

    if risco_raw:
        st.markdown("### 📊 Resultado do Modelo de ML")
        nivel = risco_raw["nivel_risco"]
        cor   = "🟢" if nivel == "BAIXO" else "🟡" if nivel == "MÉDIO" else "🔴"
        c1, c2 = st.columns(2)
        c1.metric("Probabilidade de Incidente", risco_raw["probabilidade_incidente"])
        c2.metric("Nível de Risco", f"{cor} {nivel}")

    st.divider()

    # ── Proposta do agente ─────────────────────────────────────────
    st.markdown("### 📄 Proposta de Seguro Gerada pelo Agente")
    st.markdown(proposta)

    st.divider()

    # ── Log do ReAct ──────────────────────────────────────────────
    with st.expander("🔎 Ver log do loop ReAct"):
        for entry in log:
            st.markdown(f"**Passo {entry['passo']} → `{entry['ferramenta']}`**")
            st.code(entry["resultado"], language="json")