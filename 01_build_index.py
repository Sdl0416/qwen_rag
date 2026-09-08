from rag_app import build_index


if __name__ == "__main__":
    build_index(
        docs_dir="docs",
        chunk_size=300,
        overlap=50
    )