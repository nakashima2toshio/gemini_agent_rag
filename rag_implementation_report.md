## 調査報告: RAG 実装タイプと使用クラス・関数

**結論:**
このシステムの実装は **「標準的な RAG (Standard RAG)」** に **「ReAct エージェントによる動的なツール利用」** を組み合わせたものであり、Qdrant 公式等が定義する「Hybrid RAG (Dense + Sparse ベクトルを組み合わせた検索)」そのものではありません。

UI上の表記 "Gemini 2.0 Flash + ReAct + Qdrant Hybrid RAG" の "Hybrid" は、おそらく「LLMの知識」と「外部ナレッジ(Qdrant)」を使い分けるという意味、または「Gemini」と「Qdrant」のハイブリッド構成という意味で使われていると推測されます。

### 詳細分析

#### 1. RAG の実装タイプ
*   **Standard RAG (Dense Retrieval)**:
    *   `qdrant_client_wrapper.py` の `search_collection` 関数および `services/qdrant_service.py` を確認すると、単一の `query_vector` (Dense Vector) を用いて `client.search` (または `query_points`) を実行しています。
    *   キーワード検索 (Sparse Vector / BM25) との併用や、Reciprocal Rank Fusion (RRF) 等のハイブリッド検索ロジックは見当たりません。
    *   **証拠**: `models.Distance.COSINE` のみが設定されており、Sparse Vector 用の設定がありません。

#### 2. 主要なクラス・関数

この RAG システムの中核を担っているクラスと関数は以下の通りです。

| カテゴリ | ファイル | クラス / 関数 | 役割 |
| :--- | :--- | :--- | :--- |
| **Agent (ReAct)** | `ui/pages/agent_chat_page.py` | `run_agent_turn` | ユーザー入力に対し、LLMがツールを使うか判断し、実行・回答を行うメインループ。Reflection機能も含む。 |
| | `agent_tools.py` | `search_rag_knowledge_base` | LLMから呼び出されるツール。RAG検索を実行し、結果をテキスト形式で返す。 |
| **Retrieval (検索)** | `agent_tools.py` | `search_collection` (wrapper呼び出し) | Qdrant クライアントを使用してベクトル検索を実行する。 |
| | `qdrant_client_wrapper.py` | `search_collection` | Qdrant の `search` API を叩く下位レイヤー。 |
| **Embedding (ベクトル化)** | `helper_embedding.py` | `GeminiEmbedding.embed_text` | クエリをベクトル化する (Gemini API, 3072次元)。 |
| **Database (Vector DB)** | `qdrant_client_wrapper.py` | `QdrantClient` | Qdrant との通信を担当。 |

#### 3. 処理フロー概要

1.  **ユーザー入力**: `agent_chat_page.py` でユーザーが質問。
2.  **ReAct 判断**: Gemini 2.0 Flash が `SYSTEM_INSTRUCTION` に基づき、「検索が必要」と判断。
3.  **ツール実行**: `search_rag_knowledge_base(query)` が呼ばれる。
4.  **ベクトル化**: `helper_embedding.py` でクエリを Embedding 化。
5.  **検索**: `qdrant_client_wrapper.py` 経由で Qdrant を検索 (Dense Vector Search)。
6.  **回答生成**: 検索結果 (Context) を含めて Gemini が最終回答を生成。
7.  **Reflection**: 生成された回答を自己評価・修正。

### 提言

もし「Hybrid RAG (キーワード検索 + ベクトル検索)」を意図しているのであれば、Qdrant の Sparse Vector 機能 (BM25) を導入し、検索ロジックをアップデートする必要があります。現状のままであれば、表記を "ReAct RAG Agent" 等とするのが技術的に正確です。
