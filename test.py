from agent import get_agent
from langchain_core.messages import HumanMessage
from database import init_db

init_db()

config = {
    "configurable": {
        "thread_id": "test_thread_id"
    }
}

print("1. Starting application")

agent = get_agent("gemini-3.5-flash")

print("2. Agent created")

for message_chunk, metadata in agent.stream(
    {
        "messages": [
            HumanMessage(content="what is my name ?")
        ]
    },
    config=config,
    stream_mode="messages"
):
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)

# for message_chuck,metadata in agent.stream(
#     {'messages':[HumanMessage(content="Generate a blog about Machine lerning")]},

#     config=config,
#     stream_mode='messages'
# ):
#      if message_chuck.content:
#           print(message_chuck.content,end=" ",fkush=True)

   

    
    
    