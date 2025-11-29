import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# --- 環境変数読み込み (.env 内に OPENAI_API_KEY を設定) ---
load_dotenv()


# --- 専門家として回答する LLM 関数 ---
def get_llm_response(user_text: str, expert_type: str) -> str:

    # 専門家の種類（A/B）
    system_prompt_map = {
        "A": "あなたは経理の専門家です。正確で詳細な会計処理や帳簿管理に関する説明を提供してください。",
        "B": "あなたは財務の専門家です。資金調達、財務分析、投資判断など戦略的な財務管理に関する助言を提供してください。",
    }

    system_prompt = system_prompt_map.get(expert_type, "You are a helpful assistant.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text)
    ]

    # LangChain のバージョン差吸収
    try:
        response = llm.predict_messages(messages)
        return response.content
    except Exception:
        response = llm(messages)
        return response.content if hasattr(response, "content") else str(response)




# --- Streamlit UI ---
st.set_page_config(page_title="Expert LLM App", layout="centered")


st.title("LangChain × Streamlit 専門家プロンプト切替アプリ")


expert_choice = st.radio(
    "専門家の種類を選択してください：",
    ("A", "B"),
    horizontal=True
)


# --- 入力フォーム ---
user_input = st.text_area("テキストを入力してください：")
expert_choice = st.radio(
"専門家の種類を選択してください：",
("A", "B"),
horizontal=True
)


# --- 送信ボタン ---
if st.button("送信"):
    result = get_llm_response(user_input, expert_choice)
    if result:
        st.write(result)