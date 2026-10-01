def _format_chunks(docs, label: str) -> list[str]:
    return [
        f"[{label}, p.{d.metadata.get('page', '?')}] {d.page_content.strip()}"
        for d in docs if d.page_content.strip()
    ]

def make_rag_node(retriever, label: str, k: int = 5):
    def node(state: State) -> dict:
        query = state.get("rewritten_query") or state["messages"][-1].content
        programme = state.get("programme")
        docs = retriever.invoke(f"{programme}: {query}" if programme else query)
        return {"retrieved_context": _format_chunks(docs[:k], label)}
    return node

academic_rag_node = make_rag_node(academic_retriever, "Academic Handbook")
fee_rag_node = make_rag_node(fee_retriever, "Fee Structure")