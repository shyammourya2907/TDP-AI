```mermaid
graph TD
    A[User Query] --> B[Embed Query]
    B --> C[Search Vector DB]
    C --> D[Retrieve Context]
    D --> E[Send Context + Query to LLM]
    E --> F[Generate Final Answer]
```
