import os
import json
from channels.generic.websocket import AsyncWebsocketConsumer
import chromadb
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from asgiref.sync import sync_to_async
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re
# Function to strip text till the last full stop
def strip_until_last_full_stop(text):
    last_index = text.rfind('.')
    if last_index != -1:
        return text[:last_index + 1]
    return text
# from langchain.chains import create_history_aware_retriever
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_retrieval_chain
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain.memory import ConversationBufferMemory
# from langchain_core.messages import HumanMessage
# from langchain.chains.combine_documents import create_stuff_documents_chain

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

hf = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

chroma_directory = r'C:/Users/Dell/Desktop/my_project/chroma_db'
vector_db = Chroma(persist_directory=chroma_directory, embedding_function=hf)

retriever = vector_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.4,
        "k": 5
    }
)


#repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=150,
    do_sample=False,
    huggingfacehub_api_token='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'
)

qa = RetrievalQA.from_chain_type(
           llm=llm,
           chain_type="stuff",
           retriever=retriever,
           verbose=False,
        )

#chat_history=[]
class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
#        global chat_history
#        chat_history = []
        print("WebSocket connection opened")
        await self.accept()

    async def disconnect(self, close_code):
        print("WebSocket connection closed with code:", close_code)

    async def receive(self, text_data):
        print("Received message:", text_data)
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        response_data = await self.process_user_query(message)
        print("Sending response:", response_data)

        await self.send(text_data=json.dumps({
            'response_message': response_data['response_message']
        }))

    @sync_to_async
    def process_user_query(self, query):

        # review_system_prompt = SystemMessagePromptTemplate(
        #     prompt=PromptTemplate(
        #         input_variables=["context"],
        #         template=review_template_str,
        #     )
        # )

        # review_human_prompt = HumanMessagePromptTemplate(
        #     prompt=PromptTemplate(
        #         input_variables=["question"],
        #         template="{question}",
        #     )
        # )
        # messages = [review_system_prompt, review_human_prompt]

        # review_prompt_template = ChatPromptTemplate(
        #     input_variables=["context", "question"],
        #     messages=messages,
        # )

        # review_chain = (
        #     {"context": retriever, "question": RunnablePassthrough()}
        #     | review_prompt_template
        #     | llm
        #     | StrOutputParser()
        # )
        # question = query
        # result = review_chain.invoke(question)
        # #print(result)
        # # Process the response to extract only the system-generated answer
        # response_text = str(result).strip()
        
        # # Assuming the response has the format "System: <answer>"
        # if response_text.startswith("Chatbot:"):
        #     response_text = response_text.replace("Chatbot:", "").strip()
        
        # print(response_text)
        # return {'response_message': response_text}

        # qa = RetrievalQA.from_chain_type(
        #    llm=llm,
        #    chain_type="stuff",
        #    retriever=retriever,
        #    verbose=False,
        # )
        # context = qa.run(query)
        # qa_system_prompt = """
        #     You are a professional and friendly chatbot AI assistant working at IndiaNIC. 
        #     Your goal is to assist users by providing accurate, concise, and contextually relevant responses to their queries. 
        #     You should respond as if you are a knowledgeable and approachable employee of IndiaNIC. 
        #     Use the following guidelines to construct your answers:

        #     1. **Professionalism:** Maintain a courteous and respectful tone in all interactions.
        #     2. **Accuracy:** Use the provided context to answer the user's question accurately. If the answer is not in the context, say that you don't know.
        #     3. **Conciseness:** Keep your responses to a maximum of one sentence.
        #     4. **Clarity:** Ensure your answers are clear and easy to understand.
        #     5. **Engagement:** Use a friendly and engaging tone, and ask follow-up questions when appropriate.

        #     Here is some context to help you answer the question:
        #     <context>
        #     {context}
        #     </context>
        #     """
        # question_answering_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             qa_system_prompt,
        #         ),
        #         MessagesPlaceholder(variable_name="messages"),
        #     ]
        # )

        # document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
        # print(document_chain.invoke(
        #     {
        #         "context": context,
        #         "messages": [
        #             HumanMessage(content=query)
        #         ],
        #     }
        # ))
        #PROMPT = PromptTemplate(template=qa_system_prompt, input_variables=[ "context"])
        
        # qa = RetrievalQA.from_chain_type(
        #    llm=llm,
        #    chain_type="stuff",
        #    retriever=retriever,
        #    verbose=False,
        #    chain_type_kwargs={"prompt": PROMPT}
        # )
        
        result = qa.run(query)
        stripped_answer = strip_until_last_full_stop(result)
        print(stripped_answer)
        return {'response_message': stripped_answer}
    

    # contextualize_q_system_prompt = """Given a chat history and the latest user question \
        # which might reference context in the chat history, formulate a standalone question \
        # which can be understood without the chat history. Do NOT answer the question, \
        # just reformulate it if needed and otherwise return it as is."""
        # contextualize_q_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", contextualize_q_system_prompt),
        #         MessagesPlaceholder("chat_history"),
        #         ("human", "{input}"),
        #     ])
        # history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        # qa_system_prompt = """You are a chatbot AI assistant at IndiaNIC.
        #     Act as a humanoid chatbot and give responses such that you are employed at IndiaNIC.
        #     Use the following pieces of retrieved context to answer the question. \
        #     If you don't know the answer, just say that you don't know. \
        #     Use three sentences maximum and keep the answer concise.\
        #     {context}"""
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", qa_system_prompt),
        #         MessagesPlaceholder("chat_history"),
        #         ("human", "{input}"),
        #     ]
        # )

        # document_chain = create_stuff_documents_chain(llm, prompt, output_parser=StrOutputParser())
        # retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        # response = retrieval_chain.invoke({"input": query, "chat_history": chat_history})
        # chat_history.append(HumanMessage(content=query))
        # chat_history.append(HumanMessage(content=response["answer"]))
        # print(response["answer"])
        # return response["answer"]
