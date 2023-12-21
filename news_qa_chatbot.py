# from dotenv import load_dotenv
# load_dotenv()
import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

from crawling import crawl_titles_presses_links
from crawling import extract_news
import re
import os

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

############################
# 앞으로 할 일
# 0. 제목으로 txt 파일 만들기(practice.py 참고)
# 1. DB 관련 RAG 방법 도입하기
# 2. 프롬프트 엔지니어링(답변에 제목, 내용, 날짜 들어가게 하기 + 반말로 답변)

def trim_name(name):
    pattern = re.compile(r'[^가-힣a-zA-Z0-9]+')
    return pattern.sub('', name)

def main():
    titles, presses, links = crawl_titles_presses_links()
    
    for i in range(len(titles)):
        name = trim_name(titles[i][:10].strip())
        news, date = extract_news(links[i])
   
        with open(f'{name}.txt', 'w') as f:
            f.write(titles[i] + "\n")
            f.write(date + "\n")
            f.write(presses[i] + "\n")
            f.write(news)
            
    # 출력 세팅
    st.set_page_config(
    page_title="AI News Chat",
    page_icon=":sunglasses:")

    st.title("_AI News :blue[QA Chat]_ :sunglasses:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        buttons = {}
        st.write('AI 관련 기사 모음')
        for i in range(7):
            buttons[f'button{i}'] = st.button(titles[i])
            if buttons[f'button{i}']:
                name = trim_name(titles[i][:10].strip())
                with open(f'{name}.txt') as f:
                    uploaded_file = f.read()
                text_chunks = get_text_chunks(uploaded_file)
                vetorestore = get_vectorstore(text_chunks)
            
                st.session_state.conversation = get_conversation_chain(vetorestore)
                st.session_state.processComplete = True
            

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 관심 있는 기사를 클릭한 후 관련 질문을 해주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": '''모든 답변은 3문장 이내로 해줘. 내가 마지막에 요약해달라고 하면 내가 제공하는 기사를 참고하여 요약해줘. 내가 제공하는 기사는 아래와 같아.

                # <기사>
                # {기사}

                # 단, 요약을 할 때는 아래 예시와 같은 구성으로 요약해줘

                # 제목 : SKB, IPTV에 AI 장착… 고객 맞춤형 쇼핑·콘텐츠 서비스 제공

                # 일시 : 2023.12.20. 오후 1:37

                # 언론사 : 중앙일보

                # 요약 내용 :
                # SK브로드밴드는 내년 상반기에 AI B tv를 선보일 예정이며, 이를 통해 고객에게 맞춤형 쇼핑과 콘텐츠 서비스를 제공할 것이다. AI 기술을 활용하여 TV 시청자를 자동으로 인식하고 개인 맞춤형 콘텐츠를 추천하는 기능도 제공할 예정이다. 또한, AI 쇼핑 서비스를 통해 영상 속 제품 정보를 확인하고 구매할 수 있으며, OTT 홈 서비스도 제공할 예정이다.

                # 참고로, 제공하는 기사에서 첫번째 줄은 제목이고, 두번째 줄은 일시, 세번째 줄은 언론사, 그 이후로는 기사 내용이야.'''+query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']

                st.markdown(response)



# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_texts(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore):
    # template = """
    # 너는 기자야. 요약해달라고 하면 내가 제공하는 기사를 참고하여 요약해줘. 내가 제공하는 기사는 아래와 같아.

    # <기사>
    # {기사}

    # 단, 요약을 할 때는 아래 예시와 같은 구성으로 요약해줘

    # 제목 : SKB, IPTV에 AI 장착… 고객 맞춤형 쇼핑·콘텐츠 서비스 제공

    # 일시 : 2023.12.20. 오후 1:37

    # 언론사 : 중앙일보

    # 요약 내용 :
    # SK브로드밴드는 내년 상반기에 AI B tv를 선보일 예정이며, 이를 통해 고객에게 맞춤형 쇼핑과 콘텐츠 서비스를 제공할 것이다. AI 기술을 활용하여 TV 시청자를 자동으로 인식하고 개인 맞춤형 콘텐츠를 추천하는 기능도 제공할 예정이다. 또한, AI 쇼핑 서비스를 통해 영상 속 제품 정보를 확인하고 구매할 수 있으며, OTT 홈 서비스도 제공할 예정이다.

    # 참고로, 제공하는 기사에서 첫번째 줄은 제목이고, 두번째 줄은 일시, 세번째 줄은 언론사, 그 이후로는 기사 내용이야.
    # """

    # # ChatGPT 모델을 로드합니다.
    # chatgpt = ChatOpenAI(temperature=0)

    # #ChatGPT에게 역할을 부여합니다.(위에서 정의한 Template 사용)
    # system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # #사용자가 입력할 매개변수 template을 선언합니다.
    # human_template = "{기사}"
    # human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # #ChatPromptTemplate에 system message와 human message 템플릿을 삽입합니다.
    # chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            verbose=True,
            # combine_docs_chain_kwargs={'prompt': QA_PROMPT} 
        )

    return conversation_chain



# if __name__ == '__main__':
main()

