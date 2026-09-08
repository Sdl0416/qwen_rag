from rag_app import ask_direct
from rag_app import ask_rag


if __name__ == "__main__":

    question = "QM-Cam X3在没有DHCP服务时使用什么IP？"

    print("=" * 60)
    print("直接使用 Qwen")
    print("=" * 60)

    direct_answer = ask_direct(question)

    print(direct_answer)


    print("\n" + "=" * 60)
    print("Qwen + RAG")
    print("=" * 60)

    rag_answer, contexts = ask_rag(
        question=question,
        top_k=3
    )

    print(rag_answer)

    print("\n检索来源：")

    for item in contexts:
        print(
            item["source"],
            item["page"],
            round(item["score"], 4)
        )
