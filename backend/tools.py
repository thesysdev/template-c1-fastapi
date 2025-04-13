import random
from pydantic import BaseModel
from openai import pydantic_function_tool

class GetWeather(BaseModel):
    city: str
    country: str

    def execute(self):
        weather = random.choice(["sunny", "cloudy", "rainy"])
        print(f"the weather in {self.city}, {self.country} is {weather}")
        return f"the weather in {self.city}, {self.country} is {weather}"
    
tool_registry = {
    "get_weather": GetWeather,
}

openai_tools = [
    pydantic_function_tool(tool, name=name)
    for name, tool in tool_registry.items()
]

def exec_tool(name: str, tool_call_args: str):
    print(f"executing tool {name} with args {repr(tool_call_args)}")
    json_args = tool_call_args.strip() if tool_call_args else "{}"
    if not json_args:
        json_args = "{}"
    tool_instance = tool_registry[name].model_validate_json(json_args)
    return tool_instance.execute()