'''
<requirements>
".env" 파일 생성하여 OPENAI API키 삽입 필요
ex) OPENAI_API_KEY=sk-000000000000

'''

# from dotenv import load_dotenv # OPENAI API키를 메인 코드에 직접적으로 표시하지 않기 위함
# load_dotenv()

import streamlit as st # wrapping하기 위함
import tiktoken # token을 세기 위함

from langchain.chains import ConversationalRetrievalChain # 메모리를 가지고 있는 chain 사용하기 위함
from langchain.chat_models import ChatOpenAI # OPENAI 사용

from langchain.text_splitter import RecursiveCharacterTextSplitter # text 나누기 위함
from langchain.embeddings import HuggingFaceEmbeddings # 한국어 특화된 임베딩 모델

from langchain.memory import ConversationBufferMemory # 몇 개까지의 대화를 메모리에 넣어줄지 결정하기 위함
from langchain.vectorstores import FAISS # 임시 벡터 저장소 구축

from langchain.callbacks import get_openai_callback # 메모리 구현을 위한 추가적인 라이브러리
from langchain.memory import StreamlitChatMessageHistory # 메모리 구현을 위한 추가적인 라이브러리

from crawling import crawl_titles_presses_links # 뉴스 제목, 언론사, 링크를 크롤링
from crawling import extract_news # 뉴스 본문 및 날짜 추출
import re # 정규표현식을 사용하기 위함


def main():
    # 크롤링한 뉴스 제목, 언론사, 링크
    titles, presses, links = crawl_titles_presses_links()
    
    
    for i in range(len(titles)):
        # 뉴스 파일 제목(뉴스 제목에서 한글, 영어, 숫자가 아닌 글자를 제외 + .txt)
        name = trim_name(titles[i][:10].strip())
        
        # 뉴스 본문 및 날짜 추출
        news, date = extract_news(links[i])

        # 뉴스 제목, 날짜, 언론산, 뉴스 본문 저장
        with open(f'{name}.txt', 'w') as f:
            f.write(titles[i] + "\n")
            f.write(date + "\n")
            f.write(presses[i] + "\n")
            f.write(news)
            
    # 어플리케이션 탭 제목 및 아이콘 설정
    st.set_page_config(
    page_title="AI News Chat",
    page_icon=":sunglasses:")

    # 어플리케이션 제목
    st.title("_AI News :blue[QA Chat]_ :sunglasses:")

    # st.session_state 변수 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    # 어플리케이션의 사이드바
    with st.sidebar:
        # 기사 버튼을 담을 딕셔너리
        buttons = {}
        st.write('AI 관련 기사 모음')
        for i in range(7):
            # 기사 버튼 설정
            buttons[f'button{i}'] = st.button(titles[i])
            
            # 특정 기사 버튼 누르는 경우, 해당 기사 파일 불러오기/텍스트 쪼개기/벡터화
            if buttons[f'button{i}']:
                name = trim_name(titles[i][:10].strip())
                with open(f'{name}.txt') as f:
                    uploaded_file = f.read()
                text_chunks = get_text_chunks(uploaded_file)
                vetorestore = get_vectorstore(text_chunks)

                # 벡터 저장소를 활용하여 LLM이 답변을 할 수 있도록 chain 구성
                st.session_state.conversation = get_conversation_chain(vetorestore) 
                
            
    # 채팅의 초깃값 설정
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 관심 있는 기사를 클릭한 후 질문을 해주세요!"}]

    # 특정 role(user 또는 assistant)의 컨테이너에 해당 content 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 대화 메모리를 위한 설정
    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        # 사용자 질문을 session_state.messages에 저장
        st.session_state.messages.append({"role": "user", "content": query})
        # 어플리케이션에 사용자의 질문 출력
        with st.chat_message("user"):
            st.markdown(query)
            
        # 어플리케이션에 챗봇 답변 출력
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            
            # 답변 나올 때까지 spinner 실행
            with st.spinner("Thinking..."):
                # LLM의 답변
                result = chain({"question": '''모든 답변은 2문장 이내로 해줘. 내가 요약해달라고 요청하면 아래의 예시를 참고하여 같은 구조로 요약해줘.

                        - 제목 : 
                        - 일시 : 
                        - 언론사 : 
                        - 요약 내용 :
                        
                    '''+query})
                
                # 답변에 담겨 있는 '채팅을 주고 받은 내용'을 session_state.chat_history 변수에 저장
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                    
                # 답변을 response에 저장
                response = result['answer']

                # 답변 출력
                st.markdown(response)

        # 챗봇 답변을 session_state.messages에 저장
        st.session_state.messages.append({"role": "assistant", "content": response})


# 한글, 영어, 숫자가 아닌 글자를 제외하는 함수
def trim_name(name):
    pattern = re.compile(r'[^가-힣a-zA-Z0-9]+')
    return pattern.sub('', name)


# 토큰 개수 세는 함수(토큰을 기준으로 텍스트를 쪼개기 위한 함수)
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base") # OPEAAI의 LLM을 사용하기 때문에 'cl100k_base' 토크나이저 사용
    tokens = tokenizer.encode(text)
    return len(tokens)


# 여러 개의 chunk 나누는 함수
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# chunk를 벡터화
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'}, # streamlit의 클라우드에는 GPU 없음
                                        encode_kwargs={'normalize_embeddings': True} # 사용자의 질문과 벡터 저장소의 벡터를 비교하기 위함
                                        )  
    vectordb = FAISS.from_texts(text_chunks, embeddings)
    return vectordb


# LLM 모델을 불러와 하나의 chain으로 만드는 함수
def get_conversation_chain(vetorestore):
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',temperature=0) # RAG 시스템 구축하기에 temperature은 0으로 설정

    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'), # 답변만 history에 담도록 설정
            get_chat_history=lambda h: h # 메모리에 들어온 그대로 chat_history에 넣겠다는 의미
        )

    return conversation_chain

# 실행
main()

