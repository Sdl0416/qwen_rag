from rag_app import ask_rag


if __name__ == "__main__":

    print("===== 企业知识库问答系统 =====")
    print("输入 exit 退出")

    while True:

        question = input("\n请输入问题：")

        if question.lower() == "exit":
            break

        answer, contexts = ask_rag(
            question=question,
            top_k=3
        )

        print("\n回答：")
        print(answer)

        print("\n来源：")

        for item in contexts:
            print(
                "-",
                item["source"],
                "页码:",
                item["page"],
                "相似度:",
                round(item["score"], 4)
            )
