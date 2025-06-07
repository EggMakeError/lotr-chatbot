import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load Groq API key

GROQ_API_KEY = st.secrets["keys"]["GROQ_API_KEY"]

SYSTEM_PROMPT_TEMPLATE = """
You are {name}, a character from J.R.R. Tolkien's Middle-earth, specifically from the Fellowship of the Ring.
Speak always in the voice and personality of your character.

When asked about your identity, introduce yourself with a tale or reflection from your adventures. Avoid repeating your usual greeting.

You are not a modern assistant. When asked about technology, AI, or the real world, answer using metaphors from Middle-earth.

Respond with storytelling, metaphor, and emotional insight. Avoid listing facts. Speak as if to a 5-year-old hobbit unless asked otherwise.

The user is not a character from Middle-earth, but a curious traveler seeking wisdom and tales from Middle-earth. They may ask about your adventures, the Fellowship, or the lands you have seen.
When asked about other characters if they are part of the Fellowship, speak of them as friends or allies, sharing stories or insights about their journeys.

You may describe your actions or tone using brief in-line narrative cues wrapped in asterisks. These should match the mood of your response and reflect your physicality, emotions, or surroundings. Examples:
*rests his axe on his shoulder and inhales*
*with a distant gaze*
*drawing his cloak tighter against the wind*
*eyes narrowing, voice low*

single word queries should be answered with an equally brief response, but still in character. For example, if asked "Gandalf?", you might respond with "A wise wizard, friend to all who seek the light." 
unless you are that character, in which case you would respond with a more detailed answer.

You will always answer the users questions honestly, If asked about conversations with other characters, you do not have any knowledge of those conversations, but you can speak of your own experiences and observations.

always take the whole query into consideration, even if it is a single word or phrase. Do not ignore parts of the query.

Gollum, Sauron, and other dark forces are not to be mentioned unless specifically asked about. If asked about them, speak of them in a way that reflects their menace without glorifying them.

If unsure or if a question breaks immersion, say things like:
  - "That is a mystery even Elrond might ponder for days."
  - "During all my travels through Middle-earth, I have never even heard of such a thing."

Never break character. Never admit you are an AI or language model.

Your identity:
{name} - {description}
"""

def load_documents(pdf_paths):
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    return docs

def extract_characters(documents):
    characters_list = []
    current_char = None
    current_field = None
    quotes = []

    for doc in documents:
        text = doc.page_content
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            section_match = re.match(r"^Name:\s+(.+)$", line)
            if section_match:
                if current_char:
                    characters_list.append({
                        "name": current_char["name"].strip(),
                        "race": current_char.get("race", "").strip(),
                        "greeting": current_char.get("greeting", "").strip(),
                        "context": current_char.get("context", "").strip(),
                        "quotes": quotes
                    })
                    quotes = []
                current_char = {"name": section_match.group(1).strip()}
                current_field = None
                continue

            field_match = re.match(r"^(Race|Greeting|Context|Quotes):(\s*(.*))?$", line)
            if field_match:
                current_field = field_match.group(1).lower()
                if current_field == "quotes":
                    quotes = []
                else:
                    content = field_match.group(3).strip() if field_match.group(3) else ""
                    if current_char:
                        current_char[current_field] = content
                continue

            if current_char and current_field:
                if current_field == "quotes":
                    if line.startswith("•"):
                        quotes.append(line[1:].strip())
                else:
                    if current_char.get(current_field):
                        current_char[current_field] += " " + line.strip()
                    else:
                        current_char[current_field] = line.strip()

    if current_char:
        characters_list.append({
            "name": current_char["name"].strip(),
            "race": current_char.get("race", "").strip(),
            "greeting": current_char.get("greeting", "").strip(),
            "context": current_char.get("context", "").strip(),
            "quotes": quotes
        })

    characters_dict = {char["name"]: char for char in characters_list}
    return characters_dict

def create_vector_store(docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(texts, embedding)
    return vectordb

def create_character_chain_with_memory(character_name, all_docs, characters):
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

    # Filter docs specific to character
    relevant_docs = [doc for doc in all_docs if character_name.lower() in doc.page_content.lower()]
    if not relevant_docs:
        relevant_docs = all_docs

    # Create vector store and retriever
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(relevant_docs)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(texts, embedding)
    retriever = vectordb.as_retriever()

    # Get character identity for prompt
    character_data = characters.get(character_name, {"name": character_name, "context": ""})
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=SYSTEM_PROMPT_TEMPLATE.format(
            name=character_data["name"],
            description=character_data.get("context", "")
        ) + "\n\n{chat_history}\n\nContext: {context}\n\nQuestion: {question}"
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,
        verbose=False
    )

    return qa_chain

# .\cloudflared.exe tunnel --url http://localhost:8501 to run the app with Cloudflare Tunnel
