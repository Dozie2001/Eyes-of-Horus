"""
Embedding provider system for StangWatch clip search.

Converts text (AI reasons + visual descriptions) into vectors for
similarity search in Supabase pgvector.

Supports multiple providers — user picks one in config.yaml.
All providers implement the same interface: text in, vector out.

Usage:
    from agent.embeddings import build_embedding_provider

    provider = build_embedding_provider(config)
    vector = provider.embed("someone standing near the gate with a bag")
    vectors = provider.embed_batch(["text1", "text2"])
"""

import logging
from typing import Optional, Protocol

import httpx

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Interface all embedding providers implement."""

    def embed(self, text: str) -> Optional[list[float]]:
        """Embed a single text. Returns vector or None on failure."""
        ...

    def embed_batch(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Embed multiple texts. Returns list of vectors or None on failure."""
        ...

    def is_available(self) -> bool:
        """Check if the provider is reachable."""
        ...

    @property
    def dimensions(self) -> int: ...

    @property
    def model_name(self) -> str: ...


class JinaEmbedding:
    """Jina AI — recommended default. 1M tokens/month free, 768 dims."""

    API_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(self, api_key: str, model: str = "jina-embeddings-v3",
                 dims: int = 768):
        self._api_key = api_key
        self._model = model
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return f"jina/{self._model}"

    def embed(self, text: str) -> Optional[list[float]]:
        try:
            response = httpx.post(
                self.API_URL,
                json={"model": self._model, "input": [text]},
                headers={"Authorization": f"Bearer {self._api_key}",
                          "Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"Jina embed failed: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> Optional[list[list[float]]]:
        try:
            response = httpx.post(
                self.API_URL,
                json={"model": self._model, "input": texts},
                headers={"Authorization": f"Bearer {self._api_key}",
                          "Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()["data"]
            return [item["embedding"] for item in data]
        except Exception as e:
            logger.warning(f"Jina batch embed failed: {e}")
            return None

    def is_available(self) -> bool:
        return bool(self._api_key)


class OllamaEmbedding:
    """Ollama local — unlimited, free, offline. 768 dims with nomic-embed-text."""

    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "nomic-embed-text", dims: int = 768):
        self._host = host.rstrip("/")
        self._model = model
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    def embed(self, text: str) -> Optional[list[float]]:
        try:
            response = httpx.post(
                f"{self._host}/api/embed",
                json={"model": self._model, "input": text},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["embeddings"][0]
        except Exception as e:
            logger.warning(f"Ollama embed failed: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> Optional[list[list[float]]]:
        results = []
        for text in texts:
            vec = self.embed(text)
            if vec is None:
                return None
            results.append(vec)
        return results

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self._host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


class HuggingFaceEmbedding:
    """HuggingFace Inference API — 1,000 req/day free. 384 dims with bge-small."""

    BASE_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/"

    def __init__(self, api_key: str, model: str = "BAAI/bge-small-en-v1.5",
                 dims: int = 384):
        self._api_key = api_key
        self._model = model
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return f"huggingface/{self._model}"

    def embed(self, text: str) -> Optional[list[float]]:
        try:
            response = httpx.post(
                f"{self.BASE_URL}{self._model}",
                json={"inputs": text, "options": {"wait_for_model": True}},
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=60,
            )
            response.raise_for_status()
            # HF returns nested array for sentence-transformers — take mean
            data = response.json()
            if isinstance(data[0], list):
                # Token-level embeddings — mean pool
                import statistics
                n_tokens = len(data)
                n_dims = len(data[0])
                return [statistics.mean(data[t][d] for t in range(n_tokens))
                        for d in range(n_dims)]
            return data
        except Exception as e:
            logger.warning(f"HuggingFace embed failed: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> Optional[list[list[float]]]:
        results = []
        for text in texts:
            vec = self.embed(text)
            if vec is None:
                return None
            results.append(vec)
        return results

    def is_available(self) -> bool:
        return bool(self._api_key)


class CohereEmbedding:
    """Cohere — 100 calls/min free. 1024 dims with embed-english-v3.0."""

    API_URL = "https://api.cohere.com/v2/embed"

    def __init__(self, api_key: str, model: str = "embed-english-v3.0",
                 dims: int = 1024):
        self._api_key = api_key
        self._model = model
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return f"cohere/{self._model}"

    def _embed(self, texts: list[str], input_type: str) -> Optional[list[list[float]]]:
        try:
            response = httpx.post(
                self.API_URL,
                json={
                    "model": self._model,
                    "texts": texts,
                    "input_type": input_type,
                    "embedding_types": ["float"],
                },
                headers={"Authorization": f"Bearer {self._api_key}",
                          "Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["embeddings"]["float"]
        except Exception as e:
            logger.warning(f"Cohere embed failed: {e}")
            return None

    def embed(self, text: str, input_type: str = "search_document") -> Optional[list[float]]:
        result = self._embed([text], input_type)
        return result[0] if result else None

    def embed_batch(self, texts: list[str]) -> Optional[list[list[float]]]:
        return self._embed(texts, "search_document")

    def embed_query(self, text: str) -> Optional[list[float]]:
        """Cohere uses different input_type for queries vs documents."""
        return self.embed(text, input_type="search_query")

    def is_available(self) -> bool:
        return bool(self._api_key)


class GoogleEmbedding:
    """Google Generative AI — 1,500 req/day free. 768 dims."""

    def __init__(self, api_key: str, model: str = "text-embedding-004",
                 dims: int = 768):
        self._api_key = api_key
        self._model = model
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return f"google/{self._model}"

    def embed(self, text: str) -> Optional[list[float]]:
        try:
            response = httpx.post(
                f"https://generativelanguage.googleapis.com/v1/models/{self._model}:embedContent",
                json={"content": {"parts": [{"text": text}]}},
                params={"key": self._api_key},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["embedding"]["values"]
        except Exception as e:
            logger.warning(f"Google embed failed: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> Optional[list[list[float]]]:
        results = []
        for text in texts:
            vec = self.embed(text)
            if vec is None:
                return None
            results.append(vec)
        return results

    def is_available(self) -> bool:
        return bool(self._api_key)


class VoyageEmbedding:
    """Voyage AI — 50M tokens/month free. 512 dims with voyage-3-lite."""

    API_URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str = "voyage-3-lite",
                 dims: int = 512):
        self._api_key = api_key
        self._model = model
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return f"voyage/{self._model}"

    def embed(self, text: str) -> Optional[list[float]]:
        try:
            response = httpx.post(
                self.API_URL,
                json={"model": self._model, "input": [text]},
                headers={"Authorization": f"Bearer {self._api_key}",
                          "Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"Voyage embed failed: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> Optional[list[list[float]]]:
        try:
            response = httpx.post(
                self.API_URL,
                json={"model": self._model, "input": texts},
                headers={"Authorization": f"Bearer {self._api_key}",
                          "Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()["data"]
            return [item["embedding"] for item in data]
        except Exception as e:
            logger.warning(f"Voyage batch embed failed: {e}")
            return None

    def is_available(self) -> bool:
        return bool(self._api_key)


class OpenAIEmbedding:
    """OpenAI — paid only. Configurable dims with text-embedding-3-small."""

    API_URL = "https://api.openai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small",
                 dims: int = 768):
        self._api_key = api_key
        self._model = model
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"

    def embed(self, text: str) -> Optional[list[float]]:
        try:
            response = httpx.post(
                self.API_URL,
                json={"model": self._model, "input": text,
                      "dimensions": self._dims},
                headers={"Authorization": f"Bearer {self._api_key}",
                          "Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"OpenAI embed failed: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> Optional[list[list[float]]]:
        try:
            response = httpx.post(
                self.API_URL,
                json={"model": self._model, "input": texts,
                      "dimensions": self._dims},
                headers={"Authorization": f"Bearer {self._api_key}",
                          "Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()["data"]
            return [item["embedding"] for item in data]
        except Exception as e:
            logger.warning(f"OpenAI batch embed failed: {e}")
            return None

    def is_available(self) -> bool:
        return bool(self._api_key)


def build_embedding_provider(config) -> Optional[EmbeddingProvider]:
    """
    Build the configured embedding provider from StangWatchConfig.

    Returns None if no provider can be configured (missing API key, etc).
    """
    emb_cfg = config.embeddings
    secrets = config.secrets
    provider = emb_cfg.provider.lower()
    dims = emb_cfg.dimensions

    if provider == "jina":
        key = secrets.jina_api_key
        if not key:
            logger.warning("Jina API key not set, embedding disabled")
            return None
        return JinaEmbedding(api_key=key, model=emb_cfg.model, dims=dims)

    elif provider == "ollama":
        host = emb_cfg.ollama_host or config.agent.ollama_host
        return OllamaEmbedding(host=host, model=emb_cfg.model, dims=dims)

    elif provider == "huggingface":
        key = secrets.huggingface_api_key
        if not key:
            logger.warning("HuggingFace API key not set, embedding disabled")
            return None
        return HuggingFaceEmbedding(api_key=key, model=emb_cfg.model, dims=dims)

    elif provider == "cohere":
        key = secrets.cohere_api_key
        if not key:
            logger.warning("Cohere API key not set, embedding disabled")
            return None
        return CohereEmbedding(api_key=key, model=emb_cfg.model, dims=dims)

    elif provider == "google":
        key = secrets.google_api_key
        if not key:
            logger.warning("Google API key not set, embedding disabled")
            return None
        return GoogleEmbedding(api_key=key, model=emb_cfg.model, dims=dims)

    elif provider == "voyage":
        key = secrets.voyage_api_key
        if not key:
            logger.warning("Voyage API key not set, embedding disabled")
            return None
        return VoyageEmbedding(api_key=key, model=emb_cfg.model, dims=dims)

    elif provider == "openai":
        key = secrets.openrouter_api_key  # reuse if set, or dedicated
        # Check for dedicated OpenAI key first
        openai_key = secrets._env.get("OPENAI_API_KEY", "")
        key = openai_key or key
        if not key:
            logger.warning("OpenAI API key not set, embedding disabled")
            return None
        return OpenAIEmbedding(api_key=key, model=emb_cfg.model, dims=dims)

    else:
        logger.warning(f"Unknown embedding provider: {provider}")
        return None
