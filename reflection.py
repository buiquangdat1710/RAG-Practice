from typing import List, Dict

class Reflection:
    def __init__(self, llm_client):
        """
        llm_client: OpenAI client đã khởi tạo (vd: openai)
        """
        self.llm_client = llm_client

    def rewrite(self, messages: List[Dict], current_query: str) -> str:
        """
        Viết lại current_query thành câu hỏi độc lập từ context.

        :param messages: Lịch sử chat (dạng OpenAI chat messages)
        :param current_query: Câu hỏi hiện tại từ người dùng
        :return: Câu hỏi đã viết lại
        """
        # Lấy 10 messages gần nhất không phải role = system
        chat_history = [msg for msg in messages if msg['role'] in ('user', 'assistant')][-10:]

        history_text = ""
        for msg in chat_history:
            role = "Khách" if msg["role"] == "user" else "Bot"
            history_text += f"{role}: {msg['content']}\n"
        history_text += f"Khách: {current_query}\n"

        prompt = [
            {
                "role": "system",
                "content": "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question."
            },
            {
                "role": "user",
                "content": history_text
            }
        ]

        # Gọi LLM để rewrite câu hỏi
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt
        )

        rewritten = response.choices[0].message.content.strip()
        print(f"🔁 Reflection: \"{rewritten}\"")
        return rewritten
