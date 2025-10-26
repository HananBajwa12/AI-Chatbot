import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os

class Chatbot:
    def __init__(self, qa_path="qa_data.csv"):
        print("🧠 Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.qa_path = qa_path

        # Load or create knowledge base
        if os.path.exists(qa_path):
            self.qa_data = pd.read_csv(qa_path)
            print(f"✅ Knowledge base loaded with {len(self.qa_data)} entries.")
        else:
            print("⚠️ No knowledge base found, creating new one...")
            self.qa_data = pd.DataFrame(columns=["question", "answer"])

        # Precompute embeddings
        if len(self.qa_data) > 0:
            self.qa_data['embedding'] = self.qa_data['question'].apply(
                lambda x: self.model.encode(x, convert_to_tensor=True)
            )

    def get_answer(self, user_input):
        if len(self.qa_data) == 0:
            return "I don't have any knowledge yet. Please teach me by providing an answer."

        # Compute similarity
        query_embedding = self.model.encode(user_input, convert_to_tensor=True)
        scores = [util.cos_sim(query_embedding, q_emb).item() for q_emb in self.qa_data['embedding']]
        best_idx = int(torch.argmax(torch.tensor(scores)))
        best_score = max(scores)

        if best_score < 0.60:
            # Unknown question – ask for training input
            return {
                "response": "🤔 I’m not sure about that. Would you like to teach me the correct answer?",
                "needs_training": True,
                "user_question": user_input
            }
        else:
            return {
                "response": self.qa_data.iloc[best_idx]['answer'],
                "needs_training": False
            }

    def learn(self, question, answer):
        """Save a new Q&A pair and update embeddings."""
        new_row = pd.DataFrame([{"question": question, "answer": answer}])
        self.qa_data = pd.concat([self.qa_data, new_row], ignore_index=True)

        # Encode and save
        self.qa_data['embedding'] = self.qa_data['question'].apply(
            lambda x: self.model.encode(x, convert_to_tensor=True)
        )
        self.qa_data.to_csv(self.qa_path, index=False)
        print(f"🧩 Learned new info: '{question}' → '{answer}'")
        return "Got it! I’ve learned this new information."
