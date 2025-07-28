import streamlit as st
from embeddings import Embeddings
from vector_db import VectorDatabase
from semantic_router.route import Route
from semantic_router.router import SemanticRouter
from semantic_router.samples import productsSample, chitchatSample
from reflection import Reflection
import pandas as pd
import openai
import os
import numpy as np

# Hàm ghép thông tin sản phẩm
def build_combine_row(row):
    combine = f"Tên sản phẩm: {row['title']}\n"
    combine += f"Mô tả: {row['product_specs']}\n"
    combine += f"Giá: {row['current_price']}\n"
    combine += f"Ưu đãi: {row['product_promotion']}\n"
    combine += f"Màu sắc: {row['color_options']}\n"
    return combine

# Khởi tạo session_state để lưu hội thoại
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": """Bạn là một nhân viên tư vấn bán hàng chuyên nghiệp tại cửa hàng Quang Đạt Phone. Xưng em và xưng khách hàng là anh/chị. Đôi khi sử dụng icon emoji trong câu trả lời. Nhiệm vụ của bạn là trả lời các câu hỏi của khách hàng một cách rõ ràng, thân thiện và dựa hoàn toàn vào các thông tin sản phẩm được cung cấp bên dưới.
Chỉ sử dụng thông tin có trong dữ liệu. Không tự tạo ra thông tin nếu không được cung cấp.
Nếu không tìm thấy câu trả lời, hãy lịch sự trả lời rằng hiện tại bạn chưa có đủ thông tin để tư vấn chính xác.
Hãy ưu tiên ngắn gọn, dễ hiểu. Nếu khách hỏi gợi ý sản phẩm, hãy liệt kê một vài mẫu phù hợp và lý do tại sao nên chọn.
Luôn giữ thái độ lịch sự, chuyên nghiệp và hỗ trợ hết mình."""}
        ]
    if "history" not in st.session_state:
        st.session_state.history = []

# Hàm load dữ liệu và setup
@st.cache_resource(show_spinner="🔄 Đang load dữ liệu...")
def setup():
    df = pd.read_csv("hoanghamobile.csv")
    df['information'] = df.apply(build_combine_row, axis=1)

    vector_db = VectorDatabase(db_type="mongodb")
    embedding = Embeddings(model_name="text-embedding-3-small", type="openai")

    routes = [
        Route(name="products", samples=productsSample),
        Route(name="chitchat", samples=chitchatSample)
    ]
    router = SemanticRouter(embedding, routes)

    if vector_db.count_documents("products") == 0:
        for _, row in df.iterrows():
            embedding_vector = embedding.encode(row['information'])
            vector_db.insert_document(
                "products",
                {
                    "title": row['title'],
                    "embedding": embedding_vector,
                    "information": row['information']
                }
            )
    return embedding, vector_db, router

# Hàm xử lý truy vấn người dùng
def handle_query(query, embedding, vector_db, router):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    reflection = Reflection(openai)

    rewritten_query = reflection.rewrite(st.session_state.messages, query)
    route_result = router.guide(rewritten_query)
    best_route = route_result[1]

    st.chat_message("assistant").markdown(f"**[Định tuyến]:** `{best_route}`")

    if best_route == "uncertain":
        return "🤔 Em chưa chắc chắn về câu hỏi này. Anh/chị có thể nói rõ hơn được không?"

    elif best_route == "products":
        query_embedding = embedding.encode(rewritten_query)
        results = vector_db.query("products", query_embedding, limit=5)

        context = ""
        for r in results:
            context += f"{r['information']}\n"

        base_prompt = st.session_state.messages[0]["content"]
        st.session_state.messages[0]["content"] = base_prompt + f"\nDữ liệu sản phẩm liên quan:\n{context}"
        st.session_state.messages.append({"role": "user", "content": rewritten_query})

    else:
        st.session_state.messages.append({"role": "user", "content": query})

    # Gọi OpenAI chat dạng stream
    response_stream = embedding.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages,
        stream=True
    )

    assistant_reply = ""
    placeholder = st.empty()
    for chunk in response_stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            assistant_reply += token
            placeholder.markdown(assistant_reply + "▌")

    placeholder.markdown(assistant_reply)
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    return None

# ---------- Giao diện Streamlit ----------
st.set_page_config(page_title="Tư vấn Quang Đạt Phone", page_icon="📱", layout="centered")
st.title("📞 Chatbot Tư vấn Quang Đạt Phone")

init_session()
embedding, vector_db, router = setup()

for msg in st.session_state.messages[1:]:  # Bỏ system prompt
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Anh/chị muốn hỏi gì về sản phẩm?")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    handle_query(user_input, embedding, vector_db, router)
