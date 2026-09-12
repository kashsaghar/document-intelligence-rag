from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def main():
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    word1, word2 = "strategy", "tactics"
    vector1 = embedding_function.embed_query(word1)
    vector2 = embedding_function.embed_query(word2)

    print(f"Vector for '{word1}': {vector1[:5]}... (length {len(vector1)})")
    print(f"Vector for '{word2}': {vector2[:5]}... (length {len(vector2)})")

    similarity = cosine_similarity(vector1, vector2)
    print(f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    main()
