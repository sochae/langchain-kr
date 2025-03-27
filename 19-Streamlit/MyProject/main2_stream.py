import streamlit as st 
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

st.title("나만의 챗GPT 👍")


# 처음 1번만 실행하기 위한 코드 
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위해 생성
    st.session_state["messages"] = []   # [("user", "안녕하세요!"), ("assistant", "안녕하세요!")]


# 사이드바 생성 
with st.sidebar:
    # 초기화 버튼
    clear_btn = st.button("대화 초기화")


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


# 초기화 버튼이 눌리면
if clear_btn:
        # st.write("버튼이 눌렸습니다.")
        st.session_state["messages"] = []


# 이전 대화기록 출력
print_messages()


# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")


# 만약에 사용자 입력이 들어오면,
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)
    
    # chain 생성
    chain = create_chain()
    response = chain.stream({"question": user_input})
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