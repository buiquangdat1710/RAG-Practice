from embeddings import Embeddings
from vector_db import VectorDatabase
import pandas as pd

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

    inserted_count = 0
    for index, row in df.iterrows():
        title = row['title']
        if not vector_db.document_exists("products", {"title": title}):
            doc = row['information']
            embedding_vector = embedding.encode(doc)
            vector_db.insert_document(
                collection_name="products",
                document={
                    "title": title,
                    "embedding": embedding_vector,
                    "information": doc
                }
            )
            inserted_count += 1
            print(f"Inserted document {index + 1}/{len(df)}: {title}")
    if inserted_count == 0:
        print("All documents already exist in the vector database, skipping insertion.")
    else:
        print(f"Inserted {inserted_count} new documents.")

    # Query + RAG
    query = "Có điện thoại đắt nhất bên bạn là gì, có ưu đãi gì không ?"
    query_embedding = embedding.encode(query)
    results = vector_db.query(
        collection_name="products",
        query_vector=query_embedding,
        limit=7
    )
    print("Thông tin được tìm kiếm thây:")
    for result in results:
        print(f"Title: {result['title']}, Information: {result['information']}")
        print("-" * 50)
    
    # Prompt LLM -> LLM answer query based on RAG
    prompt = f"Trả lời câu hỏi dựa trên thông tin sau:\n{query}\n\n"
    for result in results:
        prompt += f"Thông tin: {result['information']}\n"
    prompt += "Trả lời: "
    print(prompt)

    answer = embedding.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Bạn là một nhân viên tư vấn bán hàng chuyên nghiệp tại cửa hàng Quang Đạt Phone. Xưng em và xưng khách hàng là anh/chị. Đôi khi sử dụng icon emoji trong câu trả lời. Nhiệm vụ của bạn là trả lời các câu hỏi của khách hàng một cách rõ ràng, thân thiện và dựa hoàn toàn vào các thông tin sản phẩm được cung cấp bên dưới.
Chỉ sử dụng thông tin có trong dữ liệu. Không tự tạo ra thông tin nếu không được cung cấp.
Nếu không tìm thấy câu trả lời, hãy lịch sự trả lời rằng hiện tại bạn chưa có đủ thông tin để tư vấn chính xác.
Hãy ưu tiên ngắn gọn, dễ hiểu. Nếu khách hỏi gợi ý sản phẩm, hãy liệt kê một vài mẫu phù hợp và lý do tại sao nên chọn.
Luôn giữ thái độ lịch sự, chuyên nghiệp và hỗ trợ hết mình."""},
            {"role": "user", "content": prompt}
        ]
    )
    print("LLM Answer:", answer.choices[0].message.content.strip())

if __name__ == "__main__":
    main()
