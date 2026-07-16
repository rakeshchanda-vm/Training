"""Vendored, langchain_community-free ports of the vector/lexical backends.

These modules were lifted from ``langchain_community`` (which is being sunset)
and stripped of their ``langchain_community`` dependencies so andromeda only
relies on ``langchain_core`` plus the native SDKs (rank-bm25, faiss-cpu,
opensearch-py).
"""
