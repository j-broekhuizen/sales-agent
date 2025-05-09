from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
import json
import os


class InputState(TypedDict):
    question: str


class OutputState(TypedDict):
    answer: str


class OverallState(InputState, OutputState):
    pass


def load_sales_data():
    # Get the absolute path to the sales data.json file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sales_data_path = os.path.join(current_dir, "sales_data.json")

    # Load the sales data from the JSON file
    with open(sales_data_path, "r") as file:
        sales_data = json.load(file)

    return sales_data


def answer_node(state: InputState):
    # Load the sales data
    sales_data = load_sales_data()

    # Create a sales expert prompt
    sales_expert_prompt = """You are an expert sales analyst with deep knowledge of sales performance metrics and team management.
    You have access to a dataset of sales representatives containing information about:
    - Sales representative names
    - Sales quotas (target goals)
    - Current attainment (shown as decimal percentages of quota)
    
    Please analyze this data to provide helpful, accurate, and insightful answers to sales-related questions.
    Identify top performers, underperforming representatives, and trends in the data.
    
    The user's question is: {question}
    """

    llm = ChatOpenAI(model="gpt-4o-mini")

    response = llm.invoke(
        sales_expert_prompt.format(question=state["question"])
        + "\n\nHere is the sales data to reference: "
        + json.dumps(sales_data, indent=2)
    )

    # Return the answer along with the original question
    return {"answer": response.content, "question": state["question"]}


# Build the graph with explicit schemas
builder = StateGraph(OverallState, input=InputState, output=OutputState)
builder.add_node(answer_node)
builder.add_edge(START, "answer_node")
builder.add_edge("answer_node", END)
graph = builder.compile()
