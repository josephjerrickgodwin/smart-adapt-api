import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from src.service.llm.hf_client import hf_client
from src.service.prompts.agentic_rag_prompts import (
    decision_prompt,
    strategy_prompt,
    rewrite_prompt,
    sufficiency_prompt,
)
from src.service.rag_service import RAGService
from src.service.utils.misc_service import get_messages_content

log = logging.getLogger(__name__)


@dataclass
class RAGDecision:
    """Represents the decision made by the agentic RAG system."""
    use_rag: bool
    confidence: float
    reasoning: str


@dataclass
class SearchStrategy:
    """Represents the search strategy for each iteration."""
    query: str
    top_k: int
    reasoning: str


@dataclass
class RAGMemory:
    """Represents the accumulated memory from RAG searches."""
    context: str
    sources: List[Dict[str, Any]]
    confidence: float
    is_sufficient: bool
    reasoning: str


class AgenticRAGService:
    """
    Agentic RAG service that intelligently decides when to use RAG,
    performs iterative searches, and builds comprehensive memory.
    """
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.max_iterations = 2
        self.confidence_threshold = 0.7
        self.sufficiency_threshold = 0.8
        self.max_top_k = 20  # Maximum number of results to retrieve
        self.last_heartbeat = 0  # Track last heartbeat time
        
    async def decide_rag_usage(
        self, 
        history: List[Dict[str, Any]], 
        query: str
    ) -> RAGDecision:
        """
        Determine whether RAG is needed for the given query and history.
        
        Args:
            history: Conversation history
            query: Current user query
            
        Returns:
            RAGDecision with use_rag flag, confidence, and reasoning
        """
        # Create a prompt for the decision-making model
        decision_prompt_str = self._create_decision_prompt(history, query)
        
        try:
            # Use the LLM to make the decision
            response = await self._get_llm_decision(decision_prompt_str)
            decision_data = self._parse_decision_response(response)
            
            return RAGDecision(
                use_rag=decision_data.get("use_rag", False),
                confidence=decision_data.get("confidence", 0.0),
                reasoning=decision_data.get("reasoning", "No reasoning provided")
            )
        except Exception as e:
            log.error(f"Error in RAG decision making: {e}")
            # Fallback: use RAG if query contains specific keywords
            use_rag = self._fallback_rag_decision(query)
            return RAGDecision(
                use_rag=use_rag,
                confidence=0.5,
                reasoning=f"Fallback decision due to error: {str(e)}"
            )
    
    async def decide_search_strategy(
        self,
        original_query: str,
        accumulated_context: str,
        history: List[Dict[str, Any]],
        iteration: int,
        previous_queries: List[str]
    ) -> SearchStrategy:
        """
        Let the model decide the search strategy (query and top-k) for the current iteration.
        
        Args:
            original_query: The original user query
            accumulated_context: Context accumulated so far
            history: Conversation history
            iteration: Current iteration number
            previous_queries: List of queries used in previous iterations
            
        Returns:
            SearchStrategy with query, top_k, and reasoning
        """
        strategy_prompt_str = self._create_strategy_prompt(
            original_query, accumulated_context, history, iteration, previous_queries
        )
        
        try:
            # Use the LLM to decide the search strategy
            response = await self._get_llm_strategy(strategy_prompt_str)
            strategy_data = self._parse_strategy_response(response)
            
            # Validate and constrain top_k
            top_k = min(max(strategy_data.get("top_k", 5), 1), self.max_top_k)
            
            return SearchStrategy(
                query=strategy_data.get("query", original_query),
                top_k=top_k,
                reasoning=strategy_data.get("reasoning", "No reasoning provided")
            )
        except Exception as e:
            log.error(f"Error in search strategy decision: {e}")
            # Fallback: use original query with default top_k
            return SearchStrategy(
                query=original_query,
                top_k=5,
                reasoning=f"Fallback strategy due to error: {str(e)}"
            )
    
    async def rewrite_query(self, query: str, history: List[Dict[str, Any]]) -> str:
        """Rewrite the user query based on recent conversation history for clarity and specificity."""
        history_text = get_messages_content(history[-3:]) if history else 'No history'
        rewrite_prompt_str = rewrite_prompt(query, history_text)
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that rewrites user queries for clarity and retrieval."},
                {"role": "user", "content": rewrite_prompt_str}
            ]
            response = ""
            for chunk in hf_client.stream(messages=messages):
                if chunk:
                    response += chunk
            # Take the first non-empty line as the rewritten query
            rewritten = response.strip().splitlines()[0] if response.strip() else query
            return rewritten
        except Exception as e:
            log.error(f"Error rewriting query: {e}")
            return query

    async def build_memory(
        self, 
        query: str, 
        history: List[Dict[str, Any]],
        event_emitter=None
    ) -> RAGMemory:
        """
        Build comprehensive memory through iterative RAG searches.
        
        Args:
            query: Current user query
            history: Conversation history
            
        Returns:
            RAGMemory with accumulated context and metadata
        """
        accumulated_context = ""
        all_sources = []
        iteration = 0
        previous_queries = []
        
        while iteration < self.max_iterations:
            iteration += 1

            # Calculate the progress
            progress = int((iteration / self.max_iterations) * 100)
            
            if event_emitter:
                try:
                    await event_emitter({
                        "type": "status",
                        "data": {
                            "action": "thinking",
                            "description": f"Examining the question - {progress}%",
                            "done": False
                        }
                    })
                except Exception as e:
                    log.error(f"Failed to emit status event in iteration {iteration}: {e}")

            # --- Query rewriting step ---
            try:
                rewritten_query = await self.rewrite_query(query, history)
                log.info(f"Iteration {iteration} rewritten query: {rewritten_query}")
            except Exception as e:
                log.error(f"Failed to rewrite query in iteration {iteration}: {e}")
                rewritten_query = query  # Fallback to original query
                
            if event_emitter:
                try:
                    await event_emitter({
                        "type": "status",
                        "data": {
                            "action": "thinking",
                            "description": f"Building a search strategy - {progress}%",
                            "done": False
                        }
                    })
                except Exception as e:
                    log.error(f"Failed to emit status event in iteration {iteration}: {e}")
                    
            log.info(f"Agentic RAG iteration {iteration}")
            
            # Let the model decide the search strategy
            try:
                search_strategy = await self.decide_search_strategy(
                    original_query=rewritten_query,
                    accumulated_context=accumulated_context,
                    history=history,
                    iteration=iteration,
                    previous_queries=previous_queries
                )
            except Exception as e:
                log.error(f"Failed to decide search strategy in iteration {iteration}: {e}")

                # Fallback strategy
                search_strategy = SearchStrategy(
                    query=rewritten_query,
                    top_k=5,
                    reasoning=f"Fallback strategy due to error: {str(e)}"
                )

            log.info(f"Iteration {iteration} strategy: query='{search_strategy.query}', top_k={search_strategy.top_k}")
            log.info(f"Strategy reasoning: {search_strategy.reasoning}")
            
            # Check if this query has been used before
            if search_strategy.query in previous_queries:
                log.warning(f"Query '{search_strategy.query}' already used, skipping iteration {iteration}")
                break
            
            # Add to previous queries
            previous_queries.append(search_strategy.query)
            
            # Perform RAG search with the decided strategy
            try:
                search_results = await self.rag_service.search(
                    query=search_strategy.query,
                    k=search_strategy.top_k,
                    return_embeddings=False
                )
            except Exception as e:
                log.error(f"Failed to perform RAG search in iteration {iteration}: {e}")
                search_results = []

            if not search_results:
                log.warning(f"No search results found in iteration {iteration}")
                break
            
            # Extract context from search results
            context_chunk = self._extract_context_from_results(search_results)
            accumulated_context += f"\n\n--- Information {iteration} ---\n{context_chunk.strip() or ''}"
            
            # Add sources to the collection
            log.info(f"Step {iteration}: Adding sources to the collection")
            all_sources.extend([
                {
                    "content": result.get("label", ""),
                    "metadata": result.get("metadata", {}),
                    "distance": result.get("score", 0.0),
                    "iteration": iteration,
                    "query": search_strategy.query,
                    "top_k": search_strategy.top_k
                }
                for result in search_results
            ])
            
            # Check if we have sufficient information
            if event_emitter:
                try:
                    await event_emitter({
                        "type": "status",
                        "data": {
                            "action": "thinking",
                            "description": f"Thinking about a solution - {progress}%",
                            "done": False
                        }
                    })
                except Exception as e:
                    log.error(f"Failed to emit status event in iteration {iteration}: {e}")
                    
            try:
                sufficiency_check = await self._check_sufficiency(
                    query, accumulated_context, history
                )
            except Exception as e:
                log.error(f"Failed to check sufficiency in iteration {iteration}: {e}")
                sufficiency_check = {"is_sufficient": False, "confidence": 0.5, "reasoning": f"Error in sufficiency check: {str(e)}"}
            
            if sufficiency_check["is_sufficient"]:
                if event_emitter:
                    try:
                        await event_emitter({
                            "type": "status",
                            "data": {
                                "action": "thinking",
                                "description": f"A decision has been made",
                                "done": False
                            }
                        })
                    except Exception as e:
                        log.error(f"Failed to emit status event in iteration {iteration}: {e}")
                log.info(f"Sufficient information found after {iteration} iterations")
                break
        
        # Final sufficiency assessment
        if event_emitter:
            try:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "action": "thinking",
                        "description": "Crafting a solution",
                        "done": False
                    }
                })
            except Exception as e:
                log.error(f"Failed to emit final status event: {e}")
                
        try:
            final_sufficiency = await self._check_sufficiency(
                query, accumulated_context, history
            )
        except Exception as e:
            log.error(f"Failed to perform final sufficiency check: {e}")
            final_sufficiency = {"is_sufficient": True, "confidence": 0.7, "reasoning": "Fallback sufficiency assessment"}
        
        return RAGMemory(
            context=accumulated_context.strip(),
            sources=all_sources,
            confidence=final_sufficiency["confidence"],
            is_sufficient=final_sufficiency["is_sufficient"],
            reasoning=final_sufficiency["reasoning"]
        )
    
    async def process_agentic_rag(
        self, 
        history: List[Dict[str, Any]], 
        query: str,
        event_emitter=None
    ) -> Tuple[bool, Optional[RAGMemory]]:
        """
        Main entry point for agentic RAG processing.
        
        Args:
            history: Conversation history
            query: Current user query
            
        Returns:
            Tuple of (use_rag, memory) where memory is None if RAG not needed
        """
        # Step 1: Decide whether to use RAG
        if event_emitter:
            try:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "action": "thinking",
                        "description": "Thinking about the question",
                        "done": False
                    }
                })
            except Exception as e:
                log.error(f"Failed to emit initial status event: {e}")

        # Step 1: Rewrite the query
        try:
            rewritten_query = await self.rewrite_query(query, history)
        except Exception as e:
            log.error(f"Failed to rewrite query: {e}")
            rewritten_query = query  # Fallback to original query

        # Step 2: Decide whether to use RAG
        try:
            decision = await self.decide_rag_usage(history, rewritten_query)
        except Exception as e:
            log.error(f"Failed to decide RAG usage: {e}")
            # Fallback decision
            decision = RAGDecision(
                use_rag=True,
                confidence=0.5,
                reasoning=f"Fallback decision due to error: {str(e)}"
            )
        
        if not decision.use_rag:
            log.info(f"RAG not needed for query: {query[:100]}...")
            return False, None
        
        # Step 3: Build memory through iterative searches
        log.info(f"Building RAG memory for query: {query[:100]}...")
        if event_emitter:
            try:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "action": "thinking",
                        "description": "Diving deep into the question",
                        "done": False
                    }
                })
            except Exception as e:
                log.error(f"Failed to emit thinking status event: {e}")
                
        try:
            memory = await self.build_memory(query, history, event_emitter=event_emitter)
        except Exception as e:
            log.error(f"Failed to build memory: {e}")

            # Return a minimal memory object to prevent complete failure
            memory = RAGMemory(
                context="",
                sources=[],
                confidence=0.3,
                is_sufficient=False,
                reasoning=f"Memory building failed: {str(e)}"
            )
        
        return True, memory
    
    def _create_decision_prompt(self, history: List[Dict[str, Any]], query: str) -> str:
        """Create an enhanced prompt for the LLM to decide whether RAG is needed."""
        history_text = get_messages_content(history[-5:]) if history else "No history"
        return decision_prompt(history_text, query)
    
    def _create_strategy_prompt(
        self,
        original_query: str,
        accumulated_context: str,
        history: List[Dict[str, Any]],
        iteration: int,
        previous_queries: List[str]
    ) -> str:
        """Create a prompt for the LLM to decide search strategy."""
        history_text = get_messages_content(history[-3:]) if history else "No history"
        return strategy_prompt(original_query, accumulated_context, history_text, iteration, previous_queries)
    
    async def _get_llm_decision(self, prompt: str) -> str:
        """Get decision from LLM."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that makes decisions about RAG usage."},
            {"role": "user", "content": prompt}
        ]
        
        response = ""
        for chunk in hf_client.stream(messages=messages):
            response += chunk or ''
            
        return response
    
    async def _get_llm_strategy(self, prompt: str) -> str:
        """Get search strategy from LLM."""
        messages = [
            {"role": "system", "content": "You are an expert at deciding search strategies for information retrieval."},
            {"role": "user", "content": prompt}
        ]
        
        response = ""
        for chunk in hf_client.stream(messages=messages):
            response += chunk or ''
            
        return response
    
    def _parse_decision_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract decision data."""
        try:
            # Find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
                
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        except Exception as e:
            log.error(f"Error parsing decision response: {e}")
            return {"use_rag": False, "confidence": 0.0, "reasoning": "Failed to parse response"}
    
    def _parse_strategy_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract strategy data."""
        try:
            # Find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
                
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        except Exception as e:
            log.error(f"Error parsing strategy response: {e}")
            return {"query": "", "top_k": 5, "reasoning": "Failed to parse response"}
    
    def _fallback_rag_decision(self, query: str) -> bool:
        """Fallback decision logic based on keywords."""
        rag_keywords = [
            "document", "file", "report", "data", "information", "specific",
            "latest", "current", "recent", "update", "news", "research",
            "study", "analysis", "statistics", "numbers", "figures"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in rag_keywords)

    @staticmethod
    def _extract_context_from_results(results: List[Dict[str, Any]]) -> str:
        """Extract and format context from search results."""
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result.get("label", "")
            metadata = result.get("metadata", {})
            
            # Add source information if available
            source_info = ""
            if metadata:
                source_info = f" (Source: {metadata.get('source', 'Unknown')})"
            
            context_parts.append(f"{i}. {content}{source_info}")
        
        return "\n".join(context_parts)
    
    async def _check_sufficiency(
        self, 
        query: str, 
        context: str, 
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check if the accumulated context is sufficient to answer the query."""
        sufficiency_prompt_str = sufficiency_prompt(query, context)
        try:
            messages = [
                {"role": "system", "content": "You are an expert at evaluating information sufficiency."},
                {"role": "user", "content": sufficiency_prompt_str}
            ]
            response = ""
            for chunk in hf_client.stream(messages=messages):
                response += chunk or ''

            # Parse response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > 0:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return {
                    "is_sufficient": result.get("is_sufficient", False),
                    "confidence": result.get("confidence", 0.0),
                    "reasoning": result.get("reasoning", "No reasoning provided")
                }
        except Exception as e:
            log.error(f"Error in sufficiency check: {e}")

        # Fallback: consider sufficient if we have substantial context
        is_sufficient = len(context.strip()) > 500
        return {
            "is_sufficient": is_sufficient,
            "confidence": 0.6 if is_sufficient else 0.3,
            "reasoning": f"Fallback assessment: {'Sufficient' if is_sufficient else 'Insufficient'} context length"
        }
