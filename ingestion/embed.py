"""Stage 2: create embeddings for every chunk and store them in Postgres.

Only chunks without an embedding are processed, so re-runs are quick.
Model: BAAI/bge-m3 (multilingual, 1024 numbers per text).
"""
import os
from pathlib import Path

import psycopg
import torch
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16


def pick_device() -> str:
    """Use the Mac's GPU (Apple 'mps') if available, else the CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME, device=pick_device())
    model.max_seq_length = 512   # our chunks are at most ~300 words
    return model


def main():
    load_dotenv(ROOT / ".env")
    model = load_model()
    print(f"Model {MODEL_NAME} loaded on {model.device}")

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)   # teach psycopg the VECTOR type

        rows = conn.execute(
            "SELECT id, embed_text FROM chunks WHERE embedding IS NULL ORDER BY id"
        ).fetchall()
        print(f"{len(rows)} chunks need embeddings")

        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            vectors = model.encode(
                [text for _, text in batch],
                normalize_embeddings=True,   # length 1, so cosine = dot product
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE chunks SET embedding = %s WHERE id = %s",
                    [(vec, chunk_id) for (chunk_id, _), vec in zip(batch, vectors)],
                )
            conn.commit()   # save each batch, so a crash keeps earlier work
            print(f"  embedded {min(start + BATCH_SIZE, len(rows))}/{len(rows)}")

        # HNSW index: fast approximate nearest-neighbour search
        conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING hnsw (embedding vector_cosine_ops)
        """)
        conn.commit()

        left = conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
        print(f"Done. Chunks still without embeddings: {left}")


if __name__ == "__main__":
    main()
