from embeddings import Embeddings
from vector_db import VectorDatabase
import pandas as pd
from semantic_router.route import Route
from semantic_router.router import SemanticRouter
from semantic_router.samples import productsSample
from semantic_router.samples import chitchatSample
from reflection import Reflection
import openai
import os
from rerank import Reranker
from elasticsearch_db import ElasticSearchDB


def build_combine_row(row):
    combine = f"Tên sản phẩm: {row['title']}\n"
    combine += f"Mô tả: {row['product_specs']}\n"
    combine += f"Giá: {row['current_price']}\n"
    combine += f"Ưu đãi: {row['product_promotion']}\n"
    combine += f"Màu sắc: {row['color_options']}\n"
    return combine

def main():
    df = pd.read_csv("hoanghamobile.csv")
    df['information'] = df.apply(build_combine_row, axis=1)

    vector_db = VectorDatabase(db_type="mongodb")
    embedding = Embeddings(model_name="text-embedding-3-small", type="openai")
    routes = [
        Route(name="products", samples=productsSample),
        Route(name="chitchat", samples=chitchatSample)
    ]
    router = SemanticRouter(embedding, routes)
    es_db = ElasticSearchDB()
    if es_db.count_documents() == 0:
        print("🔄 Chưa có dữ liệu trong Elasticsearch, đang chèn...")
        for index, row in df.iterrows():
            doc = {
                "title": row["title"],
                "information": row["information"]
            }
            es_db.insert_document(doc)
            print(f"Inserted document {index + 1}/{len(df)}: {row['title']}")
        print("✅ Đã chèn xong dữ liệu.")
    else:
        print("✅ Dữ liệu đã tồn tại trong Elasticsearch, bỏ qua insert.")

    print("Hệ thống đã sẵn sàng. Bạn có thể hỏi về sản phẩm. Gõ 'quit' để thoát.")
    
    # Lưu toàn bộ lịch sử hội thoại
    messages = [
        {
            "role": "system",
            "content": """Bạn là một nhân viên tư vấn bán hàng chuyên nghiệp tại cửa hàng Quang Đạt Phone. Xưng em và xưng khách hàng là anh/chị. Đôi khi sử dụng icon emoji trong câu trả lời. Nhiệm vụ của bạn là trả lời các câu hỏi của khách hàng một cách rõ ràng, thân thiện và dựa hoàn toàn vào các thông tin sản phẩm được cung cấp bên dưới.
Chỉ sử dụng thông tin có trong dữ liệu. Không tự tạo ra thông tin nếu không được cung cấp.
Nếu không tìm thấy câu trả lời, hãy lịch sự trả lời rằng hiện tại bạn chưa có đủ thông tin để tư vấn chính xác.
Hãy ưu tiên ngắn gọn, dễ hiểu. Nếu khách hỏi gợi ý sản phẩm, hãy liệt kê một vài mẫu phù hợp và lý do tại sao nên chọn.
Luôn giữ thái độ lịch sự, chuyên nghiệp và hỗ trợ hết mình."""
        }
    ]

    while True:
        query = input("💬 Câu hỏi của bạn: ")
        if query.strip().lower() in ["quit", "exit"]:
            print("Tạm biệt anh/chị! Hẹn gặp lại 😊")
            break

        # Phân loại bằng Semantic Router
        openai.api_key = os.getenv("OPENAI_API_KEY")
        reflection = Reflection(openai)
        rewritten_query = reflection.rewrite(messages, query)
        route_result = router.guide(rewritten_query)
        best_route = route_result[1]
        print(route_result)
        print(f"[Semantic Router] → Phân loại: {best_route}")

        if best_route == "uncertain":
            # Có thể hỏi lại user hoặc fallback sang chitchat
            print("🤔 Em không chắc chắn câu hỏi này. Anh/chị có thể nói rõ hơn được không?")
            continue

        elif best_route == "products":
            # RAG with Elasticsearch
            results = es_db.search(query)
            for i, result in enumerate(results):
                print(f"🔍 Kết quả {i + 1}: {result['title']}")
                print(f"   Thông tin: {result['information'][:100]}...")
            context = "\n".join([doc["information"] for doc in results])


            new_system_content = messages[0]["content"] + f"\nDữ liệu sản phẩm liên quan:\n{context}"
            messages[0]["content"] = new_system_content
            messages.append({"role": "user", "content": query})
        else:
            messages.append({"role": "user", "content": query})

        response = embedding.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = response.choices[0].message.content.strip()
        print("🤖 Trả lời:", reply)
        print("-" * 80)

        messages.append({"role": "assistant", "content": reply})


    
if __name__ == "__main__":
    main()
