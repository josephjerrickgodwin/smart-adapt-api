# Prompts for Agentic RAG Service

def decision_prompt(history_text: str, query: str) -> str:
    return f"""
You are an expert AI assistant responsible for deciding whether to use Retrieval-Augmented Generation (RAG) to answer a user's query.

**Your capabilities and limitations:**
- Your knowledge is limited to your training data, which only goes up to your knowledge cutoff date (e.g., September 2023).
- You do NOT have access to private, user-specific, or proprietary data unless it is provided in the current context.
- You cannot access real-time information or recent events unless explicitly provided.

**Your task:**
Given the user's query and recent conversation history, decide if RAG is needed to answer the query accurately and comprehensively.

**Consider the following factors:**
1. **Recency:** Does the query require up-to-date or real-time information?
2. **Specificity:** Is the query about a specific document, file, report, or dataset?
3. **Ambiguity:** Is the query ambiguous or context-dependent, requiring clarification or external information?
4. **User Intent:** Is the user asking for personal, organizational, or proprietary information?
5. **General Knowledge:** Can the query be answered with general world knowledge, reasoning, or creativity?
6. **Personalization:** Does the query require knowledge of the user's history, preferences, or private data?

**Step-by-step reasoning:**
- Analyze the query for keywords or phrases indicating the need for external or recent information.
- Consider if the answer is likely to be missing, outdated, or incomplete without RAG.
- If unsure, err on the side of using RAG for factual, technical, or context-specific queries.

**Respond with a JSON object in this exact format:**
{{
    "use_rag": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Step-by-step explanation of your decision.",
    "missing_if_no_rag": "What information would be missing or less accurate if RAG is not used?"
}}

**Examples:**
- Query: "What's the weather like today?" → use_rag: true (requires real-time info), missing_if_no_rag: "Current weather data"
- Query: "Tell me a joke." → use_rag: false (general/creative), missing_if_no_rag: "None"
- Query: "What are the latest features in Python 3.12?" → use_rag: true (recent technical info), missing_if_no_rag: "Latest Python 3.12 features"
- Query: "Summarize the Q4 report in my uploaded file." → use_rag: true (specific document), missing_if_no_rag: "Q4 report content"
- Query: "Who won the 2022 FIFA World Cup?" → use_rag: false (well-known fact), missing_if_no_rag: "None"
- Query: "What is my account balance?" → use_rag: true (personal data), missing_if_no_rag: "User's account balance"
- Query: "How do I implement a binary search in Python?" → use_rag: false (general programming knowledge), missing_if_no_rag: "None"
- Query: "What did I ask you last week?" → use_rag: true (user history), missing_if_no_rag: "User's previous queries"

**Context:**
- User Query: "{query}"
- Recent Conversation History:
{history_text}

**JSON Response:**"""

def strategy_prompt(original_query: str, accumulated_context: str, history_text: str, iteration: int, previous_queries: list) -> str:
    return f"""You are an expert at deciding search strategies for information retrieval.

Context:
- Original Query: "{original_query}"
- Current Iteration: {iteration}
- Previous Queries Used: {previous_queries}
- Accumulated Context Length: {len(accumulated_context)} characters

Recent Conversation History:
{history_text}

Task: Decide the search strategy for this iteration.

Consider:
1. What specific aspect of the query should we focus on?
2. How many results (top_k) do we need? (1-20)
3. What query will give us the most relevant new information?
4. Avoid repeating previous queries exactly

Guidelines:
- For broad topics: use higher top_k (10-15)
- For specific details: use lower top_k (3-8)
- For complex queries: use medium top_k (5-12)
- Make queries specific and focused
- Build upon accumulated context

Respond with a JSON object in this exact format:
{{
    "query": "your specific search query",
    "top_k": number between 1 and 20,
    "reasoning": "explanation of your strategy"
}}

Examples:
- Broad query: "machine learning" → top_k: 12, query: "machine learning algorithms types"
- Specific query: "Python decorators" → top_k: 5, query: "Python decorator syntax examples"
- Complex query: "React performance" → top_k: 8, query: "React optimization techniques"

JSON Response:"""

def rewrite_prompt(query: str, history_text: str) -> str:
    return f"""
You are an expert assistant. Given the user's original query and the recent conversation history, rewrite the query to be as clear, specific, and unambiguous as possible for a search engine or retrieval system.

Original Query: "{query}"
Recent History: {history_text}

Rewritten Query:
"""

def sufficiency_prompt(query: str, context: str) -> str:
    return f'''You are an expert at evaluating whether retrieved information is sufficient to answer a user query.

User Query: "{query}"

Retrieved Context:
{context}

Task: Evaluate whether the retrieved context provides sufficient information to answer the user query comprehensively and accurately.

Consider:
1. Does the context directly address the main points of the query?
2. Are there any gaps in the information that would prevent a complete answer?
3. Is the information current and relevant?
4. Would additional context significantly improve the answer quality?

Respond with a JSON object in this exact format:
{{
    "is_sufficient": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of your assessment"
}}

JSON Response:''' 