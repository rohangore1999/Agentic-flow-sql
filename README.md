# SQL Agent with LangGraph

The agent allows users to query a database using natural language, with the system handling all the complexities of converting those questions into SQL queries and providing the results.

![SQL Agent flow image](sql-agent-flow.png "SQL Agent Flow")

## Overview

The SQL Agent uses a graph-based workflow to process natural language queries and convert them into SQL. The workflow follows these steps:

1. **List Tables**: Identifies available tables in the database
2. **Get Schema**: Retrieves the database schema to understand table structures
3. **Generate Query**: Creates a SQL query based on the user's natural language question
4. **Validate Query**: Checks the query for errors and fixes them if needed
5. **Execute Query**: Runs the validated SQL query against the database
6. **Format Answer**: Presents the query results in a clear, conversational format

## Key Components for Building an Agentic Flow

### 1. State Management

```python
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

The `State` class is crucial for maintaining context throughout the agent's workflow. It stores conversation history and enables stateful reasoning across multiple steps.

#### Memory in LangGraph Agents

In the context of our SQL agent, "memory" refers to how the agent maintains state and context throughout a conversation or workflow. This is crucial for creating coherent multi-step interactions.

#### Implementation of Memory

In our SQL agent, memory is implemented through the `State` class:

```python
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

This might look simple, but it's powerful. The `State` object:

1. Persists throughout the entire execution of the agent workflow
2. Is passed from node to node in the graph
3. Contains the full conversation history in the `messages` list
4. Uses the special `add_messages` annotation to properly append new messages

#### How Memory is Used

The SQL agent uses memory in several critical ways:

#### 1. Contextual Understanding

When a user asks "How many orders are more than 300 rupees?", the agent remembers:

- That it's working with a database
- What tables are available (from the `list_tables_tool` node)
- The schema of those tables (from the `get_schema_tool` node)

#### 2. Stateful Processing

Each node in the workflow graph can:

- Read information from the state: `messages = state["messages"]`
- Add information to the state: `return {"messages": [response]}`

#### 3. Workflow Decision Making

The `should_continue` function examines the memory to decide what happens next:

```python
def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        return END
    elif last_message.content.startswith("Error: "):
        return "query_gen"
    else:
        return "correct_query"
```

#### 4. Error Recovery

When errors occur, they're recorded in the state, allowing the agent to:

- Remember what went wrong
- Adjust its approach in subsequent steps
- Provide feedback based on the history of the interaction

#### Benefits of Memory in Agents

1. **Coherence**: The agent provides responses that make sense in the context of the entire conversation
2. **Progressive refinement**: Each step builds on previous steps
3. **Error correction**: The agent can remember and learn from mistakes
4. **Context awareness**: The agent doesn't need to re-query database tables or schema repeatedly

This stateful approach is what makes the agent truly "agentic" rather than just a series of disconnected function calls. The memory provides the continuity that enables complex reasoning across multiple steps.

### 2. Tools and Tool Binding

Tools are functions that allow the agent to perform specific actions:

```python
# Database query tool
@tool
def query_to_database(query: str) -> str:
    """Execute a SQL query against the database and return the result."""
    result = db.run_no_throw(query)
    if not result:
        return "No result returned from the query. Please try again."
    return result

# Binding tools to the language model
llm_with_tools = llm.bind_tools([query_to_database])
```

The SQL Agent uses several tools:

- `sql_db_list_tables`: Lists available tables in the database
- `sql_db_schema`: Retrieves database schema information
- `query_to_database`: Executes SQL queries against the database

### 3. Specialized Response Formatting

```python
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user's question.")

llm_with_final_answer = llm.bind_tools([SubmitFinalAnswer])
```

This structured output ensures consistent formatting of responses to users.

### 4. Node Functions

Each node in the graph represents a specific function:

```python
def first_tool_call(state: State):
    """Initiates the first tool call to list tables"""
    return {"messages": [AIMessage(content="", tool_calls=[{"name":"sql_db_list_tables", "args":{}, "id": "tool_call_id"}])]}

def llm_get_schema(state: State):
    """Gets the database schema using the language model"""
    messages = state["messages"]
    response = llm_to_get_schema.invoke(messages)
    return {"messages": [response]}

def generation_query(state: State):
    """Generates SQL queries based on user input"""
    message = query_generator.invoke(state)
    # Error handling for hallucinated tool calls
    # ...
    return {"messages": [message] + tool_messages}
```

### 5. Error Handling

```python
def handle_tool_error(state: State):
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            ) for tc in tool_calls
        ]
    }

# Creating nodes with error handling
def create_node_from_tool_with_fallback(tools: list):
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],
        exception_key="error"
    )
```

This robust error handling allows the agent to gracefully recover from issues and provide helpful feedback.

### 6. Workflow Graph Construction

```python
workflow = StateGraph(State)

# Adding nodes
workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node("list_tables_tool", list_tables)
workflow.add_node("model_get_schema", llm_get_schema)
# ...more nodes

# Adding edges (transitions between nodes)
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
# ...more edges

# Adding conditional logic for branching
workflow.add_conditional_edges(
    "query_gen",
    should_continue,
    {END: END, "correct_query": "correct_query"}
)
```

The directed graph defines how the agent transitions between different states and actions.

### 7. Conditional Routing

```python
def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        return END
    elif last_message.content.startswith("Error: "):
        return "query_gen"
    else:
        return "correct_query"
```

This function allows the agent to make decisions about the next steps based on the current state.