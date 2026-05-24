```mermaid
graph TD
    Start --> NodeA[Agent 1]
    NodeA --> Condition{Tool Needed?}
    Condition -- Yes --> Tool[Tool Execution]
    Tool --> NodeA
    Condition -- No --> End[Final Answer]
```
