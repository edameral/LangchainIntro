from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes

load_dotenv()
model = ChatOpenAI( temperature =0.1)

'''
messages = [
    SystemMessage(content="Translate the following from English to Italian"),
    HumanMessage(content="hi"),
]
'''
system_prompt = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt), ("user", {"text"})
    ]
)


parser = StrOutputParser()
#response = model.invoke(messages)

chain = prompt_template |model | parser

app = FastAPI(
    title= "Translator App",
    version="1.0.0",
    description="Translated Chat Bot",
)

add_routes(
    app,
    chain,
    path= "/chain"
           )

if __name__ == '__main__':
    #response = model.invoke(messages)
    #print(parser.invoke(messages))
    #print(chain.invoke({"language": "Italian", "text" : "Hi"}))

    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 8000)

