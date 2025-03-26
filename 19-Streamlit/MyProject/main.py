import streamlit as st 
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

st.title("나만의 챗GPT")


# 처음 1번만 실행하기 위한 코드 
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위해 생성
    st.session_state["messages"] = []   # [("user", "안녕하세요!"), ("assistant", "안녕하세요!")]


# 이전 대화를 출력
# for role, message in st.session_state["messages"]:
#     st.chat_message(role).write(message)
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인 생성
def create_chain():
    # prompt | llm | output_parser
    # 프롬프트
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절한 AI 어시스턴트입니다."),
            ("user", "#Question:\n{question}"),
        ]
    )
    
    # GPT
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # 출력 파서
    output_parser = StrOutputParser()
    
    # 체인 생성
    chain = prompt | llm | output_parser
    return chain


print_messages()


# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")


# 만약에 사용자 입력이 들어오면,
if user_input:
    # 화면에 대화를 출력
    # st.write(f"사용자 입력: {user_input}")
    # with st.chat_message("user"):
    #     st.write(user_input)
    
    # 사용자의 입력
    st.chat_message("user").write(user_input)
    
    # chain 생성
    chain = create_chain()
    ai_answer = chain.invoke({"question": user_input})
    
    # AI의 답변
    # st.chat_message("assistant").write(user_input)
    st.chat_message("assistant").write(ai_answer)

    # 대화 기록을 저장
    # ChatMessage(role="user", content=user_input)
    # ChatMessage(role="assistant", content=user_input)
    # st.session_state["messages"].append(("user", user_input))
    # st.session_state["messages"].append(("assistant", user_input))
    add_message("user", user_input)
    # add_message("assistant", user_input)
    add_message("assistant", ai_answer)