MAX_PERSISTENT_STORE_TOKENS=2000
NUM_RETRIEVALS=10
NUM_HISTORY_MESSAGES=50


from datetime import datetime
from nltk.probability import FreqDist
import pickle


from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.memory import ChatMessageHistory

from templates import updatePersistentTemplate , getPersistentContextTemplate , getFinalAnswerTemplate,getCompressedTemplate

OPENAI_API_KEY=""
try:
    with open('./secrets.txt', 'r') as file:
        OPENAI_API_KEY = file.readline().strip()
except:
    print("Error : File not found.")

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY , model='gpt-4')

#-------------------LOADING THE MODEL AND DATABASES

def load_list_from_vectorstore(file_path="./store/vectorstore.pkl"):
    try:
        with open(file_path, 'rb') as file:
            loaded_list = pickle.load(file)
        return loaded_list
    except:
        return []
    
def save_list_to_vectorstore(history, file_path="./store/vectorstore.pkl"):
    
    with open(file_path, 'wb') as file:
        pickle.dump(history.messages, file)


model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model='gpt-4')
output_parser = StrOutputParser()
loaded_messages=load_list_from_vectorstore()
store = [message.content for message in loaded_messages]
vectorstore = DocArrayInMemorySearch.from_texts(
    store,
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
)
retriever = vectorstore.as_retriever()
history = ChatMessageHistory()
history.add_messages(loaded_messages)


getCompressed_chain=getCompressedTemplate|model|output_parser

#----------------------------DEFINING HELPER FUNCTIONS

def getPersistentStore(x):
    path="./store/persistentStore.txt"
    ans = ""
    try:
        with open(path , "r") as file:
            ans = file.read()

        num_tokens = FreqDist(ans).N()
        if num_tokens>MAX_PERSISTENT_STORE_TOKENS:
            ans = getCompressed_chain.invoke(ans)
        return ans
    except FileNotFoundError:
        return ""


def saveNewPersistentStore(text):
    path="./store/persistentStore.txt"
    with open(path , "w") as file:
        file.write(text)
    return text


def get_timestamp(x=""):
    return datetime.now().strftime("%m/%d/%Y %H:%M:%S")

def getHazyMemory(x):
    
    docs=retriever.get_relevant_documents(x , k=NUM_RETRIEVALS)
    retriever.add_documents([Document(metadata={"time":get_timestamp()}, page_content=("User Input : "+x))])
    ans=""
    for d in docs:
        ans += d.page_content
        ans+="\n"

    return ans
def getChatHistory(x):
    ans=""
    # print(type(history.messages[0]) , history.messages[0])
    if len(history.messages) <= NUM_HISTORY_MESSAGES:
        for m in history.messages:
            ans+=m.content
            ans+="\n"
    else:
        for i in range(1 , 11):
            ans+=history.messages[-i].content
            ans+="\n"


    return ans

def addToHumanHistory(x):
    history.add_user_message(x)
    retriever.add_documents([Document(metadata={"time":get_timestamp()}, page_content=("Human Answer : "+x))])
    return x

def addToHistory(x):
    history.add_ai_message(x)
    retriever.add_documents([Document(metadata={"time":get_timestamp()}, page_content=("AI Answer : "+x))])
    return x





#--------------------------------DEFINING THE LANGCHAINS

getUpdatedPersistent_chain = {"user_message":RunnablePassthrough(),
                              "recent_chat_history": RunnableLambda(getChatHistory),
                              "timestamp":RunnableLambda(get_timestamp),
                              "persistent_memory":RunnableLambda(getPersistentStore)
                              }|updatePersistentTemplate | model | output_parser | RunnableLambda(saveNewPersistentStore)

getPersistentContext_chain = {
    "user_message" : RunnablePassthrough(),
    "recent_chat_history": RunnableLambda(getChatHistory),
    "timestamp":RunnableLambda(get_timestamp),
    "persistent_memory":getUpdatedPersistent_chain
}| getPersistentContextTemplate | model | output_parser 

getFinalAnswer_chain = RunnableLambda(addToHumanHistory)|{
    "hazy_memory" : RunnableLambda(getHazyMemory),
    "chat_history":RunnableLambda(getChatHistory),
    "persistent_memory":getPersistentContext_chain,
    "user_input":RunnablePassthrough(),
    "timestamp":RunnableLambda(get_timestamp)
} | getFinalAnswerTemplate |model|output_parser|RunnableLambda(addToHistory)





