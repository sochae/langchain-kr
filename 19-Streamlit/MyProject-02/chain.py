from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_teddynote.prompts import load_prompt
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama


def format_doc(document_list):
    return "\n\n".join(doc.page_content for doc in document_list)


# 체인 생성
def create_chain(retriever, model_name="gpt-4o"):
    # prompt | llm | output_parser

    if model_name == "ollama":
        print("===== ollama Model =====")
        # 단계 6: 프롬프트(Prompt) 생성
        prompt = load_prompt("prompts/pdf-rag-ollama.yaml", encoding="utf-8")

        # 단계 7: 언어모델(LLM) 생성
        # llm = ChatOllama(model="gemma3:latest", temperature=0)
        llm = ChatOllama(model="gemma3:latest")
    else:
        print("===== gpt Model =====")
        # 단계 6: 프롬프트(Prompt) 생성
        prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

        # 단계 7: 언어모델(LLM) 생성
        llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        # ollama인 경우, retriever에 검색된 문서를 하나로 함쳐서 -> format_doc으로 함게 넣어줘야 답변 받을 수 있음. 
        # {"context": retriever | format_doc, "question": RunnablePassthrough()}
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
