import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_teddynote.prompts import load_prompt
from langchain_teddynote import logging
from dotenv import load_dotenv
import os
from retriever import create_retriever
from chain import create_chain

# API KEY 정보로드
load_dotenv()

# 프로젝트 추적
logging.langsmith("[Project] PDF RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


st.title("Local 모델 기반 RAG 👀")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위해 생성
    st.session_state["messages"] = (
        []
    )  # [("user", "안녕하세요!"), ("assistant", "안녕하세요!")]

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않았을 경우
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # selected_prompt = "prompts/pdf-rag.yaml"

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시에 저장 (시간이 오래 걸리는 작업을 처리할 예정... embeddings)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # retriever 생성 파일 분리
    return create_retriever(file_path)


# 체인 생성 파일 분리 
# # 체인 생성
# def create_chain(retriever, model_name="gpt-4o"):
#     # prompt | llm | output_parser

#     # 단계 6: 프롬프트 생성(Create Prompt)
#     # 프롬프트를 생성합니다.
#     prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

#     # 단계 7: 언어모델(LLM) 생성
#     # 모델(LLM) 을 생성합니다.
#     llm = ChatOpenAI(model_name=model_name, temperature=0)

#     # 단계 8: 체인(Chain) 생성
#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return chain


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정 ...)
    retriever = embed_file(uploaded_file)

    # (문서 업로드 시) chain 생성
    chain = create_chain(retriever, model_name=selected_model)
    # 생성한 chain을 session_state에 저장
    st.session_state["chain"] = chain


# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"] = []


# 이전 대화기록 출력
print_messages()


# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면,
if user_input:
    # chain 생성
    # chain = create_chain(retriever)
    # chain을 매번 생성하지 않고 session_state에서 가져옴
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)

        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화 기록을 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")
