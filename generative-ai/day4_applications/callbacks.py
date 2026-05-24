"""
Topic: Day 4 - Callback Handlers
Practical: Implementing callback handlers for custom logging.
"""
from langchain.callbacks.base import BaseCallbackHandler

# Custom Callback Handler
class MyCustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[Callback] LLM is starting... Prompt: {prompts[0]}")
        
    def on_llm_end(self, response, **kwargs):
        print(f"[Callback] LLM finished generating!")

handler = MyCustomHandler()
handler.on_llm_start({}, ["Tell me a joke."])
handler.on_llm_end({})
