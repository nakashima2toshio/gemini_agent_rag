# agent_tools.py

import os
from qdrant_client import QdrantClient
from qdrant_client_wrapper import search_collection, embed_query, QDRANT_CONFIG
from config import AgentConfig

# Initialize Client
# Assuming localhost:6333 as per config
# We use the URL from the config or default to localhost
qdrant_url = QDRANT_CONFIG.get("url", "http://localhost:6333")
client = QdrantClient(url=qdrant_url)

def list_rag_collections():
    """
    利用可能なRAGのコレクション一覧（ナレッジベースの種類）を取得します。
    ユーザーが「どのような知識があるか」「コレクション一覧を教えて」と質問した場合に使用してください。
    Returns:
        str: 利用可能なコレクション名のリスト。
    """
    print(f"\n[Tool Action] コレクション一覧を取得中...")
    try:
        # Qdrantサーバーから実際のコレクション一覧を取得
        collections_response = client.get_collections()
        collections = [c.name for c in collections_response.collections]
        
        if not collections:
            return "現在、利用可能なコレクションはありません。"
            
        # configのリストと突き合わせても良いが、ここでは実態を返す
        return "利用可能なコレクション一覧:\n" + "\n".join([f"- {c}" for c in collections])
        
    except Exception as e:
        return f"コレクション一覧の取得中にエラーが発生しました: {str(e)}"

def search_rag_knowledge_base(query: str, collection_name: str = AgentConfig.RAG_DEFAULT_COLLECTION): # Modified
    """
    Qdrantデータベースから専門的な知識を検索します。
    ユーザーが「仕様」「設定」「Wikipediaの知識」「事実確認」など、
    外部知識が必要な詳細について質問した場合にこのツールを使用してください。
    
    一般的な挨拶（「こんにちは」など）や、単純な計算、
    一般的なプログラミングの文法質問には使用しないでください。
    Args:
        query (str): 検索したいキーワードや質問文。
        collection_name (str, optional): 検索対象のQdrantコレクション名。
                                         指定しない場合、デフォルトコレクションが使用されます。
    Returns:
        str: 検索されたドキュメントの内容（質問と回答のペア）。
    """
    print(f"\n[Tool Action] RAG検索を実行中: {query} (Collection: {collection_name}) ...") # Modified
    
    # Validate collection_name (Optional: configにあるものだけに制限するかどうか)
    # ここでは柔軟性を重視し、configチェックは警告レベルにするか、あるいはスキップします。
    # 実運用ではホワイトリストチェックが推奨されますが、今回はQdrantにあるものを優先します。
    
    try:
        # 1. Embed the query
        query_vector = embed_query(query)
        
        # 2. Search
        results = search_collection(
            client=client,
            collection_name=collection_name, # Modified
            query_vector=query_vector,
            limit=AgentConfig.RAG_SEARCH_LIMIT
        )
        
        if not results:
            return "検索結果が見つかりませんでした。"
            
        # 3. Format Output
        formatted_results = []
        for i, res in enumerate(results, 1):
            score = res.get("score", 0.0)
            
            # Score Filtering
            if score < AgentConfig.RAG_SCORE_THRESHOLD:
                continue

            payload = res.get("payload", {})
            q = payload.get("question", "N/A")
            a = payload.get("answer", "N/A")
            
            formatted_results.append(f"Result {i} (Score: {score:.2f}):\nQ: {q}\nA: {a}\n")
            
        if not formatted_results:
             return "検索結果は見つかりましたが、関連性スコアが低いため採用しませんでした。"

        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"検索中にエラーが発生しました: {str(e)}"