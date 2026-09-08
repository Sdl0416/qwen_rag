from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-plus")
EMBED_MODEL = os.getenv("QWEN_EMBED_MODEL", "qwen3.7-text-embedding")
EMBED_DIM = int(os.getenv("QWEN_EMBED_DIM", "1024"))

INDEX_DIR = Path("index")
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "metadata.json"


def get_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError(
            "没有找到 DASHSCOPE_API_KEY。请复制 .env.example 为 .env 并填写。"
        )
    if not BASE_URL:
        raise RuntimeError(
            "没有找到 DASHSCOPE_BASE_URL。请在 .env 中填写 Model Studio 的 "
            "OpenAI-compatible base URL。"
        )
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def read_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"source": path.name, "page": None, "text": text}]


def read_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "source": path.name,
                    "page": i,
                    "text": text,
                }
            )
    return pages


def load_documents(doc_dir: Path) -> list[dict]:
    records = []
    for path in sorted(doc_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            records.extend(read_pdf(path))
        elif suffix in {".txt", ".md"}:
            records.extend(read_txt(path))
    return records


def chunk_text(
    text: str,
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[str]:
    """
    简化版中文 RAG：按字符长度切分。
    后续可以升级为按句子、标题或 token 切分。
    """
    text = " ".join(text.split())
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def make_chunks(
    pages: list[dict],
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    chunks = []
    chunk_id = 0
    for page in pages:
        for text in chunk_text(page["text"], chunk_size, overlap):
            chunks.append(
                {
                    "id": chunk_id,
                    "source": page["source"],
                    "page": page["page"],
                    "text": text,
                }
            )
            chunk_id += 1
    return chunks


def batched(items: list[str], batch_size: int = 10) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    vectors = []

    # 文本向量接口有每次调用条数限制，使用小批次更稳妥。
    for batch in batched(texts, batch_size=10):
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
            dimensions=EMBED_DIM,
            encoding_format="float",
        )

        # 按 index 排序，确保与输入顺序一致
        data = sorted(response.data, key=lambda x: x.index)
        vectors.extend(item.embedding for item in data)

    arr = np.asarray(vectors, dtype=np.float32)

    # IndexFlatIP + L2 normalize = cosine similarity
    faiss.normalize_L2(arr)
    return arr


def build_index(
    docs_dir: str,
    chunk_size: int = 700,
    overlap: int = 120,
):
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"找不到文档目录: {docs_path}")

    pages = load_documents(docs_path)
    if not pages:
        raise RuntimeError(
            "没有读取到 PDF/TXT/MD 文档，请把文件放进 docs/。"
        )

    chunks = make_chunks(pages, chunk_size, overlap)
    if not chunks:
        raise RuntimeError("文档切分后没有得到文本 Chunk。")

    print(f"[INFO] documents/pages records: {len(pages)}")
    print(f"[INFO] chunks: {len(chunks)}")

    client = get_client()
    vectors = embed_texts(client, [item["text"] for item in chunks])

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    metadata = {
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    META_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[DONE] FAISS index: {INDEX_PATH}")
    print(f"[DONE] metadata: {META_PATH}")


def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(
            "还没有知识库索引，请先运行 build 命令。"
        )

    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, metadata


def retrieve(
    question: str,
    top_k: int = 3,
) -> list[dict]:
    index, metadata = load_index()
    chunks = metadata["chunks"]

    client = get_client()
    q_vec = embed_texts(client, [question])

    k = min(top_k, index.ntotal)
    scores, ids = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = dict(chunks[int(idx)])
        item["score"] = float(score)
        results.append(item)
    return results


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_text = []
    for i, item in enumerate(contexts, start=1):
        page_info = f"第{item['page']}页" if item["page"] else ""
        context_text.append(
            f"[资料{i}] 来源：{item['source']} {page_info}\n"
            f"{item['text']}"
        )

    joined = "\n\n".join(context_text)

    return f"""请严格依据下面提供的知识库资料回答问题。

规则：
1. 不要使用资料之外的事实来补全答案。
2. 如果资料不足以回答，请明确说“知识库中没有足够信息”。
3. 回答尽量简洁、条理清晰。
4. 回答结尾请列出使用到的资料编号，例如：参考：[资料1][资料3]。

知识库资料：
{joined}

用户问题：
{question}
"""


def ask_rag(question: str, top_k: int = 3) -> tuple[str, list[dict]]:
    contexts = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, contexts)

    client = get_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个基于企业/技术文档回答问题的知识库助手。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content or ""
    return answer, contexts


def ask_direct(question: str) -> str:
    """用于做“直接 Qwen vs RAG”简单对比实验。"""
    client = get_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def print_sources(contexts: list[dict]):
    print("\n===== 检索来源 =====")
    for i, item in enumerate(contexts, start=1):
        page_info = f" 第{item['page']}页" if item["page"] else ""
        preview = item["text"][:160].replace("\n", " ")
        print(
            f"[{i}] {item['source']}{page_info} "
            f"score={item['score']:.4f}\n    {preview}..."
        )


def chat(top_k: int):
    print("===== Qwen + RAG 知识库问答 =====")
    print("输入 exit / quit 退出。")

    while True:
        question = input("\n问题> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        answer, contexts = ask_rag(question, top_k=top_k)
        print("\n===== 回答 =====")
        print(answer)
        print_sources(contexts)


def compare(question: str, top_k: int):
    print("===== 直接 Qwen =====")
    print(ask_direct(question))

    print("\n===== Qwen + RAG =====")
    answer, contexts = ask_rag(question, top_k=top_k)
    print(answer)
    print_sources(contexts)


def parse_args():
    parser = argparse.ArgumentParser(description="简化版 Qwen + RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="构建知识库")
    p_build.add_argument("--docs", default="docs")
    p_build.add_argument("--chunk-size", type=int, default=700)
    p_build.add_argument("--overlap", type=int, default=120)

    p_chat = sub.add_parser("chat", help="进入交互问答")
    p_chat.add_argument("--top-k", type=int, default=3)

    p_ask = sub.add_parser("ask", help="单次 RAG 问答")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=3)

    p_compare = sub.add_parser(
        "compare",
        help="对比直接 Qwen 与 RAG",
    )
    p_compare.add_argument("question")
    p_compare.add_argument("--top-k", type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "build":
        build_index(
            args.docs,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    elif args.command == "chat":
        chat(args.top_k)
    elif args.command == "ask":
        answer, contexts = ask_rag(args.question, top_k=args.top_k)
        print(answer)
        print_sources(contexts)
    elif args.command == "compare":
        compare(args.question, args.top_k)


if __name__ == "__main__":
    main()
