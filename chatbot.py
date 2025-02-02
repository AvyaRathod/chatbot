import asyncio
from g4f.client import Client
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Saves data persistently
collection = chroma_client.get_or_create_collection(name="table_building_knowledge")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Chat Client
client = Client()

def retrieve_relevant_docs(query, top_k=10):
    """Retrieves the top K most relevant text chunks from ChromaDB and ensures good formatting."""
    query_embedding = embedding_model.encode([query]).tolist()  # Encode query
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    if not results["ids"][0]:  # If no relevant docs are found
        return []
    
    retrieved_texts = [match["text"] for match in results["metadatas"][0]]

    return retrieved_texts

# **Refined System Prompt for Better Control**
system_prompt = """You are a highly knowledgeable AI assistant with access to a specific knowledge base. 
Your goal is to provide accurate, relevant answers using the retrieved information. 

- ONLY use the retrieved knowledge to answer questions.
- If the retrieved knowledge is unclear, you may attempt to summarize but do NOT guess.
- If there is NO useful information retrieved, respond: 'I'm sorry, I do not have enough information to answer that.'
- Format responses in a structured way, using bullet points or explanations when necessary.
"""

conversation_history = [{"role": "system", "content": system_prompt}]

print("Welcome to the chatbot! Send a message to get started.")
user_input = ""

async def chat_with_bot():
    global user_input
    while user_input.lower() != "stop":
        user_input = input("User: ")
        if user_input.lower() == "stop":
            print("Goodbye! Chatbot session ended.")
            break
        
        conversation_history.append({"role": "user", "content": user_input})

        # Retrieve relevant knowledge
        retrieved_docs = retrieve_relevant_docs(user_input)

        # Check if retrieval is actually useful
        if not retrieved_docs:
            bot_output = "I'm sorry, I do not have enough information to answer that."
        else:
            # Properly format knowledge for better understanding
            retrieval_context = "\n\n---\n".join(retrieved_docs)
            augmented_prompt = f"""Context:\n{retrieval_context}\n\n
            IMPORTANT: You must ONLY use the above knowledge to answer. 
            If the context does not include the answer, say: 'I'm sorry, I do not have enough information to answer that.'

            Chat History:\n""" + \
            "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in conversation_history]) + "\n\nBot:"

            # Get chatbot response
            bot_output = await get_chat_response(augmented_prompt)

        # Append bot response
        conversation_history.append({"role": "bot", "content": bot_output})

        print(bot_output)

async def get_chat_response(augmented_prompt):
    """Asynchronous function to get a response from g4f"""
    bot_output = ""
    chat_completion = client.chat.completions.create(
        model="gpt-4o", 
        max_tokens=200, temperature=0.5, top_p=1, top_k=0,
        messages=[{"role": "system", "content": augmented_prompt}],
        stream=True
    )

    for completion in chat_completion:  # Regular generator, not async
        bot_output += completion.choices[0].delta.content or ""

    return bot_output

# Run the async function properly
if __name__ == "__main__":
    asyncio.run(chat_with_bot())  # Runs the chatbot safely
