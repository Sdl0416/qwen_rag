from rag_app import ask_rag
if __name__ == "__main__":
    question = "QM-Cam X3最高分辨率是多少？"
    answer, contexts = ask_rag(
        question=question,
        top_k=3
    )
    print("=" * 60)
    print("问题：")
    print(question)
    print("\n回答：")
    print(answer)
    print("\n检索来源：")
    for i, item in enumerate(contexts, start=1):
        print(f"\n资料{i}")
        print("来源：", item["source"])
        if item["page"]:
            print("页码：", item["page"])
        print(
            "相似度：",
            round(item["score"], 4)
        )
        print(
            "内容：",
            item["text"][:300]
        )
