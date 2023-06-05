from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import openai

def main():
    load_dotenv()
    st.set_page_config(page_title="The Newton")
    st.header("The Newton Sixth Form 💬")
    
    # Set up OpenAI API credentials
    openai.api_key = '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'

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # 
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        # book_name = st.text_input("Book Name:")
        user_question = st.text_input("Ask a question about your PDF:")
        # user_question2 = " "
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            # # Query OpenAI for additional answers
            # openai_response = openai.Completion.create(
            #     engine='text-davinci-003',
            #     prompt=user_question + "from the book name " + book_name,
            #     max_tokens=3000,
            #     n=1,
            #     stop=None,
            #     temperature=0
            # )
            # openai_answer = openai_response.choices[0].text.strip()

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question, temperature=0)
                print(cb)

            # combined_answer = "Combine these two answers and rewrite them to be longer, and use the first response to be the main answer " + "response 1 : "+ response[0] + " response 2 : " + openai_answer
            # # Regenerate the answer using OpenAI
            # openai_regenerated_response = openai.Completion.create(
            #     engine='text-davinci-003',
            #     prompt=combined_answer,
            #     max_tokens=2000,
            #     n=1,
            #     stop=None,
            #     temperature=0
            # )
            # regenerated_answer = openai_regenerated_response.choices[0].text.strip()


            st.write("Answer from PDF:")
            st.write(response)
        

            st.write(docs)
            st.write(knowledge_base)


if __name__ == '__main__':
    main()
