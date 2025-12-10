![agent_1.png](assets/agent_1.png)

# Agent RAG システム

本システムは、「自律型 RAG エージェント」および統合管理プラットフォームです。
システムの特徴（ReAct + Reflection、フルスクラッチ実装、Gemini 3世代対応）です。
**。** **Streamlit ベースの UI を通じて、データの取得・ベクトル化から、Qdrant データベース管理、
そして高度なエージェント対話まで、RAG パイプライン全体を一気通貫で管理・運用することができます**

**主な特徴と技術的工夫:**

1. ReAct (Reasoning + Acting):
   　　エージェント自らが「考える（Reasoning）」と「行動する（Acting）」をループ
   　　・入力プロンプトの最適化
   　　・CoT（Chain-of-Thought)のLoop
   　　・Hybrid RAG (Dense + Sparse)の検索
   　　必要な情報が揃うまで自律的に検索ツール (search_rag_knowledge_base) を行使します。
2. Reflection (自己評価結果に基づき、最終回答 (Final Answer) を抽出：自己省察):
   　　回答を作成した後、即座に出力せず「自己評価」フェーズを実行し、回答の品質を向上。
   　　検索結果との整合性やスタイルを自ら批評し、ハルシネーション（幻覚）や誤りを修正してからユーザーに回答します。
3. フルスクラッチ実装:
   　　Gemini APIを直接利用し、柔軟な制御を実現。

- 第1部: Agent 詳細設計書(Gemini ReAct Reflection Agent対応)
- 第2部: Agent アーキテクチャ概念 (Theoretical Architecture)
- 第3部: Agent 動作シーケンス (Runtime Behavior)

![lp.png](assets/lp.png?t=1765400877187)

## 目次

1. [概要](#1-概要)
   - 1.1 [本モジュールの目的](#11-本モジュールの目的)
   - 1.2 [主な機能（7画面の概要）](#12-主な機能7画面の概要)
   - 1.3 [対応データセット](#13-対応データセット)
2. [アーキテクチャ](#2-アーキテクチャ)
   - 2.1 [システム構成図（3層アーキテクチャ）](#21-システム構成図3層アーキテクチャ)
   - 2.2 [モジュール依存関係図](#22-モジュール依存関係図)
   - 2.3 [レイヤー別役割分担表](#23-レイヤー別役割分担表)
   - 2.4 [システムアーキテクチャ図（Mermaid）](#24-システムアーキテクチャ図mermaid)
   - 2.5 [コンポーネント連携シーケンス図](#25-コンポーネント連携シーケンス図)
3. [データフロー](#3-データフロー)
   - 3.1 [エンドツーエンド処理フロー図](#31-エンドツーエンド処理フロー図)
   - 3.2 [各ステップの入出力](#32-各ステップの入出力)
   - 3.3 [ディレクトリ構造](#33-ディレクトリ構造)
4. [サービス層 & ツール層](#4-サービス層--ツール層)
   - 4.1 [dataset_service.py - データセット操作](#41-dataset_servicepy---データセット操作)
   - 4.2 [qdrant_service.py - Qdrant操作](#42-qdrant_servicepy---qdrant操作)
   - 4.3 [file_service.py - ファイル操作](#43-file_servicepy---ファイル操作)
   - 4.4 [qa_service.py - Q/A生成](#44-qa_servicepy---qa生成)
   - 4.5 [agent_tools.py - エージェント用ツール](#45-agent_toolspy---エージェント用ツール)
5. [UI層 (ui/pages/)](#5-ui層-uipages)
   - 5.1 [画面一覧と遷移](#51-画面一覧と遷移)
   - 5.2 [各ページの機能詳細](#52-各ページの機能詳細)
6. [メニュー単位の処理概要・処理方式](#6-メニュー単位の処理概要処理方式)
   - 6.1 [📖 説明](#61--説明)
   - 6.2 [🤖 エージェント対話](#62--エージェント対話)
   - 6.3 [📊 未回答ログ](#63--未回答ログ)
   - 6.4 [📥 RAGデータダウンロード](#64--ragデータダウンロード)
   - 6.5 [🤖 Q/A生成](#65--qa生成)
   - 6.6 [📥 CSVデータ登録](#66--csvデータ登録)
   - 6.7 [🗄️ Qdrantデータ管理](#67--qdrantデータ管理)
   - 6.8 [🔎 Qdrant検索](#68--qdrant検索)
7. [設定・依存関係](#7-設定依存関係)
   - 7.1 [必須環境変数](#71-必須環境変数)
   - 7.2 [依存サービス](#72-依存サービス)
   - 7.3 [主要な定数・設定値](#73-主要な定数設定値)
8. [使用方法](#8-使用方法)
   - 8.1 [起動手順](#81-起動手順)
   - 8.2 [典型的なワークフロー](#82-典型的なワークフロー)
9. [ReAct エージェント詳細設計](#9-react-エージェント詳細設計)
   - 9.1 [ReAct ループの仕組み](#91-react-ループの仕組み)
   - 9.2 [主要クラス・関数 IPO 定義](#92-主要クラス関数-ipo-定義)
   - 9.3 [システムプロンプト設計](#93-システムプロンプト設計)
   - 9.4 [シーケンス図 (Agent Turn)](#94-シーケンス図-agent-turn)

---

## 1. 概要

![agent_2.png](assets/agent_2.png)

### 1.1 本モジュールの目的

`agent_rag.py` は、**Gemini 3 (2.0 Flash)** 世代に対応したRAG（Retrieval-Augmented Generation）システムの統合管理ツールです。

**一言で言うと**: Gemini活用型RAG Q&A生成・Qdrant管理、および **ReAct型エージェント** による対話を実現する統合Streamlitアプリケーション

**役割**:

- データ取得からベクトル検索までの **RAGパイプライン全体** を管理
- **ReActエージェント** を介した、ツール利用による高度な対話機能
- **Gemini API** (`gemini-2.0-flash`, `gemini-embedding-001`) を全面的に採用し、高速・低コスト・高精度を実現


| 項目           | 内容                                            |
| -------------- | ----------------------------------------------- |
| ファイル名     | agent_rag.py                                    |
| フレームワーク | Streamlit                                       |
| 起動コマンド   | `streamlit run agent_rag.py --server.port=8500` |

### 1.2 主な機能（7画面の概要）


| 画面             | アイコン | 機能概要                                                                                                   |
| ---------------- | -------- | ---------------------------------------------------------------------------------------------------------- |
| 説明             | 📖       | システムのデータフロー・ディレクトリ構造を表示                                                             |
| エージェント対話 | 🤖       | **ReAct Agent** (Gemini 2.0) との対話。ナレッジベース検索 + **Reflection (自己推敲)** による高品質な回答。 |
| 未回答ログ       | 📊       | エージェントが回答できなかった質問のログ分析                                                               |
| RAGデータDL      | 📥       | HuggingFace/ローカルファイルからデータ取得・前処理                                                         |
| Q/A生成          | 🤖       | **Gemini 2.0 Flash** によるQ&Aペア自動生成（Celery並列処理対応）                                           |
| CSVデータ登録    | 📥       | **Gemini Embedding (3072次元)** でベクトル化・登録・コレクション統合                                       |
| Qdrantデータ管理 | 🗄️     | Qdrantコレクション内容の閲覧 (Show-Qdrant)                                                                 |
| Qdrant検索       | 🔎       | セマンティック検索単体のテスト・AI応答生成                                                                 |

### 1.3 対応データセット


| データセット    | 識別子          | 説明                                   |
| --------------- | --------------- | -------------------------------------- |
| Wikipedia日本語 | `wikipedia_ja`  | Wikipedia日本語版                      |
| CC-News         | `cc_news`       | CC-News英語ニュース                    |
| Livedoor        | `livedoor`      | Livedoorニュースコーパス               |
| カスタム        | `custom_upload` | ローカルファイル（CSV/TXT/JSON/JSONL） |

---

## 2. アーキテクチャ

### 2.1 システム構成図（3層アーキテクチャ）

```mermaid
graph TD
    subgraph Presentation [プレゼンテーション層]
        Entry["agent_rag.py"]
        Pages["ui/pages/*.py"]
        Entry --- Pages
    end

    subgraph BusinessLogic [ビジネスロジック層]
        Services["services/"]
        Tools["agent_tools.py"]
        Services --- Tools
    end

    subgraph DataAccess [データアクセス層]
        API["Gemini API (2.0 Flash / Embed)"]
        DB["Qdrant"]
    end

    Presentation --> BusinessLogic
    BusinessLogic --> DataAccess
```

### 2.2 モジュール依存関係図

```mermaid
graph LR
    Main["agent_rag.py"]

    subgraph UI_Pages ["ui/pages/"]
        InitUI["__init__.py"]
        AgentPage["agent_chat_page.py"]
        LogPage["log_viewer_page.py"]
        OtherPages["... (download, qa, etc.)"]
    end

    subgraph Logic_Layer ["Logic"]
        Tools["agent_tools.py"]
        QS["qdrant_service.py"]
        LogSvc["services/log_service.py"]
    end

    subgraph Helper_Layer ["Helpers"]
        HelperRag["helper_rag.py"]
        QdrantWrapper["qdrant_client_wrapper.py"]
    end

    Main --> InitUI
    InitUI --> AgentPage
    InitUI --> LogPage
    InitUI --> OtherPages

    AgentPage --> Tools
    AgentPage --> QS
    AgentPage --> LogSvc
    Tools --> QdrantWrapper

    OtherPages --> QS
```

### 2.3 レイヤー別役割分担表


| レイヤー             | モジュール                    | 責務                                                         |
| -------------------- | ----------------------------- | ------------------------------------------------------------ |
| **エントリポイント** | `agent_rag.py`                | アプリ起動、ルーティング                                     |
| **UI層**             | `ui/pages/agent_chat_page.py` | エージェント対話UI、ReActループ制御 (`run_agent_turn`)       |
| **ツール層**         | `agent_tools.py`              | エージェントが利用するツール群 (`search_rag_knowledge_base`) |
| **サービス層**       | `services/*.py`               | データ処理、DB操作の抽象化                                   |

### 2.4 システムアーキテクチャ図（Mermaid）

```mermaid
graph TB
    subgraph UI
        Entry[EntryPoint]
        AgentUI[Agent Chat Page]
    end

    subgraph AgentLogic
        ReAct[ReAct Engine<br>run_agent_turn]
        Tools[Agent Tools<br>search_rag / list_collections]
    end

    subgraph External
        Gemini[Gemini 2.0 Flash]
        Qdrant[Qdrant Vector DB]
    end

    Entry --> AgentUI
    AgentUI --> ReAct
    ReAct -- "Prompt + History" --> Gemini
    Gemini -- "Function Call" --> ReAct
    ReAct -- "Execute" --> Tools
    Tools -- "Search" --> Qdrant
    Qdrant -- "Documents" --> Tools
    Tools -- "Observation" --> ReAct
    ReAct -- "Observation" --> Gemini
    Gemini -- "Final Answer" --> ReAct
    ReAct --> AgentUI
```

---

## 3. データフロー

(基本構成は既存と同様。RAGデータ生成パイプラインは変更なし)

### 3.1 エンドツーエンド処理フロー図

1. データDL -> 2. 前処理 -> 3. QA生成 -> 4. 埋め込み登録 -> **5. エージェントによる活用 (Search & Answer)**

---

## 4. サービス層 & ツール層

### 4.5 agent_tools.py - エージェント用ツール

**責務**: エージェントが外部環境（Qdrant）と対話するためのインターフェースを提供。


| 関数名                      | 説明                                                                           | 関連ツール名（LLM側）       |
| --------------------------- | ------------------------------------------------------------------------------ | --------------------------- |
| `search_rag_knowledge_base` | 指定されたコレクションからクエリに関連する情報を検索する。ベクトル検索を実行。 | `search_rag_knowledge_base` |
| `list_rag_collections`      | 利用可能なQdrantコレクションの一覧を返す。                                     | `list_rag_collections`      |

---

## 5. UI層 (ui/pages/)

### 5.1 画面一覧と遷移

サイドバーのラジオボタンにより、以下の画面を切り替え。

1. **説明 (`explanation`)**
2. **エージェント対話 (`agent_chat`)**
3. **未回答ログ (`log_viewer`)**
4. **RAGデータDL (`rag_download`)**
5. **Q/A生成 (`qa_generation`)**
6. **CSVデータ登録 (`qdrant_registration`)**
7. **Qdrantデータ管理 (`show_qdrant`)**
8. **Qdrant検索 (`qdrant_search`)**

### 5.2 各ページの機能詳細

#### `agent_chat_page.py` (エージェント対話)

* **機能**: Gemini 2.0 Flash を用いたチャットインターフェース。
* **特徴**:
  * **ReActループ**: 思考(Thought)と行動(Action)の可視化。
  * **Reflection**: 回答案生成後に自己評価・修正を行い、ハルシネーションの低減とスタイル統一を実現。
  * **マルチコレクション**: 検索対象のコレクションをサイドバーで選択可能。
  * **ストリーミング**: 思考プロセスを `st.expander` 内に逐次表示。

#### `log_viewer_page.py` (未回答ログ)

* **機能**: エージェントが「回答なし」と判断したクエリの履歴を表示・分析。

---

## 6. メニュー単位の処理概要・処理方式

### 6.1 📖 説明

システム全体の概要を表示。

### 6.2 🤖 エージェント対話

ReActエージェントがユーザーの質問に対し、ツール（検索）を使って回答を作成します。

* **思考の可視化**: 「なぜその検索を行うか」という推論過程を表示。
* **ツール利用**: `search_rag_knowledge_base` を自律的に呼び出し、Qdrantから情報を取得。
* **Reflection (推敲)**: 回答案を作成した後、自己評価フェーズを実行。正確性・適切性・スタイルを推敲し、より洗練された回答を提示します。

---

## 9. ReAct + Reflection エージェント詳細設計

本システムの中核である「ハイブリッド・ナレッジ・エージェント」の詳細設計です。
参考: `doc/11_agent_react.md`

### 9.1 ReAct + Reflection の仕組み

Gemini 2.0 Flash の Function Calling 機能を利用し、以下のサイクルを回します。

1. **ReAct フェーズ (解決)**:

   * **Thought (思考)**: ユーザーの入力に対し、外部知識が必要か、どのツールを使うべきか考える。
   * **Action (行動)**: ツール (`search_rag_knowledge_base`) を呼び出すことを決定し、APIにリクエスト。
   * **Observation (観察)**: ツールを実行し、その結果（検索結果やエラー）を取得。
   * **Draft Answer (ドラフト作成)**: 観察結果に基づき、回答案を生成。
2. **Reflection フェーズ (推敲)**:

   * **Critique (批評)**: 生成されたドラフト回答に対し、検索結果（コンテキスト）との整合性やスタイルを自己評価。
   * **Revise (修正)**: 必要に応じて回答を修正し、最終回答 (Final Answer) とする。

### 9.2 主要クラス・関数 IPO 定義

#### `ui.pages.agent_chat_page.run_agent_turn`

エージェントの1ターン（ユーザー発話〜最終回答）を制御するメイン関数。


| 項目        | 内容                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input**   | `chat_session`: Gemini ChatSession<br>`user_input`: ユーザーの質問文字列                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Process** | 1.`chat_session.send_message(user_input)` を送信。<br>2. 応答に `function_call` が含まれるか確認。<br>3. **含まれる場合**:<br>　a. 思考プロセスを表示・ログ記録。<br>　b. `agent_tools` 内の対応関数を実行。<br>　c. 結果を `function_response` として Gemini に返送。<br>　d. ステップ2に戻る（ReActループ）。<br>4. **含まれない場合**:<br>　a. 現在の回答をドラフト (Draft Answer) とする。<br>5. **Reflection (推敲)**:<br>　a. ドラフト回答を `REFLECTION_INSTRUCTION` と共に送信。<br>　b. 自己評価結果に基づき、最終回答 (Final Answer) を抽出。 |
| **Output**  | `final_response_text`: 最終的な回答文字列                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

#### `agent_tools.search_rag_knowledge_base`

RAG検索を実行するツール関数。


| 項目        | 内容                                                                                                                                                                                                                                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input**   | `query`: 検索クエリ<br>`collection_name`: 検索対象コレクション名 (Optional)                                                                                                                                                                                                                             |
| **Process** | 1. Qdrantヘルスチェック。<br>2. `collection_name` の存在確認。<br>3. `embed_query` でクエリをベクトル化 (Gemini Embedding)。<br>4. `search_collection` でベクトル検索。<br>5. スコア閾値 (`AgentConfig.RAG_SCORE_THRESHOLD`) でフィルタリング。<br>6. 検索結果を LLM が理解しやすいテキスト形式に整形。 |
| **Output**  | 整形された検索結果文字列 (または`[[NO_RAG_RESULT]]`)                                                                                                                                                                                                                                                    |

# Gemini Hybrid RAG Agent - 理論と実装リファレンス

本ドキュメントは、ReAct + Reflection エージェントの理論的背景（概念図）と、ユーザーが選択可能なGeminiモデルを活用した`agent_rag.py` および関連モジュールの実装詳細を体系的にまとめたリファレンスです。

---

## 第2部: アーキテクチャ概念 (Theoretical Architecture)

Gemini エージェントの思考プロセスは、大きく2つのフェーズ（解決と推敲）で構成されています。

## 2.1 Phase 1: ReAct (試行錯誤による解決)

ReActは、**「考え（Reasoning）」ながら「行動（Acting）」し、その結果を見てまた「考える」**というプロセスです。
AIは単に回答を出力するのではなく、外部ツール（検索など）を使いながら、情報が揃うまで行動を繰り返します。

```mermaid
flowchart LR
    Start([ユーザーの依頼]) --> Thought1
    subgraph ReAct_Loop [ReActループ: 解決パート]
        Thought1[Thought: 何が必要か考える] --> Action[Action: ツール実行/検索]
        Action --> Observation[Observation: 結果を観察]
        Observation --> Decision{情報十分?}
        Decision -- No --> Thought1
    end
    Decision -- Yes --> FinalAns[Draft Answer: 回答案の生成]
```

* **Thought**: 現在の状態を分析し、次に何をすべきか計画します。
* **Action**: ツール（検索）を実行します。
* **Observation**: ツールの実行結果（検索結果）を受け取ります。

## 2.1.1 入力文字列から検索クエリ生成までの処理構造（重要）

Pythonコード側で「キーワード抽出」や「クエリ整形」を行う専用の関数は実装されていません。
Geminiモデル自体が、システムプロンプトの指示に基づき、「入力文」を解釈し、「最適な検索クエリ」へと変換（推論） しています。

1. 入力フェーズ (Pythonコード: `ui/pages/agent_chat_page.py`)

* ユーザーの行動:
  チャット画面に自然文で質問を入力します。

  > 具体例: 「実験生物学では、生物の機構を解明するためにどのような操作を加えますか？」
  >
* コード処理 (`run_agent_turn` 関数):
  この文字列はそのまま Gemini API (chat_session.send_message) に渡されます。
  同時に、システムプロンプト (SYSTEM_INSTRUCTION_TEMPLATE) によって、モデルには以下の「思考のルール」が与えられています。

  > 指示: 「Thought: [なぜ検索が必要か、どのコレクションを、どんなクエリで検索するか]」
  >

2. 生成・推論フェーズ (Gemini API 内部)

* モデルの思考 (Reasoning):
  モデルはプロンプトの指示に従い、ユーザーの意図を汲み取りつつ、検索ツール (search_rag_knowledge_base)
  に渡すべき最適な引数を考えます。

  > 思考例: 「このユーザーの質問は長い。Qdrantで正確に検索するには、助詞を省いて重要なキーワードに絞ったほうが良いだろう。」
  >
* クエリの決定 (Function Call 生成):
  モデルは思考の結果に基づき、ツールの引数 query を生成します。ここでの出力が、実際の検索クエリとなります。

  > 生成パターン例:
  > ケースA (重要語抽出)*: "実験生物学 生物の機構 操作"
  > ケースB (キーワード化)*: "実験生物学 実験操作"
  > ケースC (そのまま)*: "実験生物学では、生物の機構を解明するためにどのような操作を加えますか？"
  >

※現状のプロンプトでは「キーワードのみにせよ」という強制はないため、モデルの文脈判断によりケースA～Cのように変動します。しかし、Geminは一般的に、検索に適した形（ケースAやB）へ自発的に変換する傾向があります。

3. 伝達・実行フェーズ (Pythonコード: `agent_tools.py`)

* コード処理 (`run_agent_turn` -> `search_rag_knowledge_base`):
  Gemini API から返ってきた function_call 情報（モデルが決めたクエリ）を Python 側で受け取り、そのまま検索関数を実行します。

```python
# agent_tools.py
def search_rag_knowledge_base(query: str, ...):
    # ここに来る時点で、query は既にモデルによって
    # "実験生物学 実験操作" などに変換されている可能性がある
    return qdrant_service.search_collection_rag(query, ...)
```

まとめ


| フェーズ | 担当               | 処理内容                                 | 具体例                                  |
| :------- | :----------------- | :--------------------------------------- | :-------------------------------------- |
| 1. 入力  | agent_chat_page.py | ユーザーの自然文を受け取る               | 「実験生物学では...操作を加えますか？」 |
| 2. 変換  | Gemini (LLM)       | 文脈から「検索用クエリ」を推論・生成する | 「実験生物学 実験操作」 (ケースB)       |
| 3. 実行  | agent_tools.py     | 生成されたクエリで検索を実行する         | query="実験生物学 実験操作" で検索      |

つまり、「クエリ生成ロジック」の実体は Python コードではなく、LLM の頭脳（推論プロセス）の中 にあります。

### 2.1.2 CoT (Chain of Thought) の処理構造（重要）

ReActエージェントは、最終的な回答を出す前に、思考(Thought)と行動(Action/Tool Call)を連鎖させ、論理的に答えを導き出します。
以下は、実際の実行ログに基づく思考の連鎖プロセスです。

#### 具体的な挙動の仕組み (実行ログの追跡)

1. **初期思考 (Initial Thought)**

   * **入力**: 「実験生物学では、生物の機構を解明するためにどのような操作を加えますか？」
   * **LLMの推論**: 質問の意図を理解し、外部情報が必要か判断します。
   * **思考ログ**:
     > 🧠 Thought: [生物の機構を解明するための操作に関する質問なので、一般的な知識としてwikipediaを検索してみる。]
     >
2. **ツール実行 (Action & Observation)**

   * **LLMの行動**: 推論に基づき、適切なツールと引数を生成します。
   * **ツール呼び出し**:
     > 🛠️ Tool Call: `search_rag_knowledge_base`
     > Args: `{'collection_name': 'wikipedia_ja', 'query': '実験生物学\u3000生物機構\u3000操作'}`
     >
   * **ツールの結果 (Observation)**:
     > 📝 Tool Result: Result 1 (Score: 0.50): Q: 実験生物学では... A: 人為的に操作を加え通常と異なる条件を作り出し...
     >
3. **解決思考 (Reasoning & Draft)**

   * **LLMの推論**: 検索結果を読み、質問に答えられるか判断します。
   * **思考ログ**:
     > 🧠 Thought: [検索結果から、質問に対する回答が得られた。]
     >
   * **ドラフト回答**:
     > Answer: 社内ナレッジによると、実験生物学では...
     >
4. **推敲 (Reflection)**

   * **LLMの自己評価**:
     > 🤔 Reflection Thought: ** [自己評価: 回答は質問に直接的かつ明確に答えており...修正は不要と判断しました。]**
     >

#### まとめ


| ステップ | フェーズ        | 処理内容                 | 実際のログ要素                               |
| :------- | :-------------- | :----------------------- | :------------------------------------------- |
| **1**    | **Thought**     | 検索の必要性と戦略の立案 | `Thought: ...wikipediaを検索してみる。`      |
| **2**    | **Action**      | 検索ツールの実行         | `Tool Call: search_rag_knowledge_base`       |
| **3**    | **Observation** | 検索結果の取得           | `Tool Result: ...人為的に操作を加え...`      |
| **4**    | **Draft**       | 情報の統合と回答作成     | `Answer: 社内ナレッジによると...`            |
| **5**    | **Reflection**  | 回答の品質チェック       | `Reflection Thought: ...修正は不要と判断...` |

### 2.1.3 Reflectionフェーズ (自己省察と推敲) の処理構造（重要）

検索結果を基に一度回答を作成した後、さらに「推敲」を行うプロセスです。
これにより、回答の正確性やスタイルが、システム要件（丁寧な日本語など）に合致しているか自己評価し、必要に応じて修正します。

#### プロンプト戦略 (`REFLECTION_INSTRUCTION`)

`REFLECTION_INSTRUCTION` 定数にて、以下の観点でのチェックを指示しています。

1. **正確性 (Accuracy)**: 検索結果に基づいているか？ 幻覚 (Hallucination) はないか？
2. **適切性 (Relevance)**: ユーザーの質問に直接答えているか？
3. **スタイル (Style)**: 親しみやすく丁寧な日本語（です・ます調）か？ 箇条書き等のフォーマットは適切か？

#### 具体的な挙動の仕組み

1. **ドラフト生成フェーズ** (ReActループ終了後)

   * **LLMの思考**: 検索結果（wikipedia等）から情報を得たので、回答を作成します。
     > **思考例 (Thought)**: 「検索結果から、質問に対する回答が得られた。」
     >
   * **回答案 (Draft)**:
     > 「社内ナレッジによると、実験生物学では、生物に備わっている機構を解明するために、人為的に操作を加え通常と異なる条件を作り出し、その後の変化を観察・観測します。例えば、突然変異の誘発や遺伝子導入、移植実験などを行います。」
     >
   * **コード処理 (`run_agent_turn` 後半)**: この回答案を一時変数 `final_response_text` に保持します。
2. **推敲フェーズ** (Reflection)

   * **コード処理**: `REFLECTION_INSTRUCTION` (評価プロンプト) とドラフト回答を結合し、再度 Gemini に送信します。

     > **指示**: 「以下の基準で客観的に評価し...修正してください...思考プロセスは Thought: で始めてください。」
     >
   * **LLMの思考 (Reflection Thought)**: プロンプトに従い、自分の回答を評価します。

     > **思考例**: 「[自己評価: 回答は質問に直接的かつ明確に答えており、正確性、適切性、スタイルにも問題ないため、修正は不要と判断しました。]」
     >
   * **最終回答の生成 (Final Answer)**: 評価に基づき、最終版を出力します。

#### まとめ


| フェーズ    | 担当                 | 処理内容                             | 具体例                                                 |
| :---------- | :------------------- | :----------------------------------- | :----------------------------------------------------- |
| 1. 推敲指示 | `agent_chat_page.py` | ドラフト回答 + 評価プロンプトを送信  | `REFLECTION_INSTRUCTION` + 「社内ナレッジによると...」 |
| 2. 自己評価 | `Gemini (LLM)`       | 基準（正確性・スタイル）に従って評価 | 「自己評価: ...修正は不要と判断しました。」            |
| 3. 最終化   | `Gemini (LLM)`       | 修正版（またはそのまま）の回答を出力 | 「社内ナレッジによると...（最終回答）」                |

## 2.2 Phase 2: Reflection (自己省察と推敲)

Reflectionは、生成された回答（ドラフト）に対して客観的な批評を行い、品質を高めるプロセスです。

```mermaid
flowchart LR
    Input([Draft Answer]) --> Reflect
    subgraph Reflection_Loop [Reflectionループ: 推敲パート]
        Reflect[Reflect: 批評・チェック] --> Check{問題なし?}
        Check -- No --> Revise[Revise: 修正版作成]
        Revise --> Reflect
    end
    Check -- Yes --> Output([Final Answer])
```

* **Reflect**: 正確性、適切性、スタイルをチェックします。
* **Revise**: 問題があれば修正し、最終回答を生成します。

## 2.3 統合モデル (ReAct + Reflection)

「動く（Action）」フェーズと「考える（Reflection）」フェーズを連携させることで、より高度な成果物を生み出します。

```mermaid
flowchart TD
    User([ユーザーの依頼]) --> P1
    subgraph Phase1 [Phase 1: ReAct Loop]
        direction TB
        Reasoning[思考と行動の繰り返し] --> Draft[ドラフト回答の作成]
    end

    P1 --> P2

    subgraph Phase2 [Phase 2: Reflection Loop]
        direction TB
        Draft --> Critique[ドラフトと依頼を比較・批評]
        Critique --> Revise[修正と洗練]
    end

    Revise --> Final([Final Answer: 最終回答])
```

---

# 第3部: 実装詳細 (Implementation Details)

上面的理論が、実際のPythonコードでどのように実装されているか解説します。

## 2.4 エージェント制御: `ui/pages/agent_chat_page.py`

エージェントのライフサイクル管理を行うメインコントローラーです。

* **`setup_agent(selected_collections, model_name)` 関数**:
  * **役割**: エージェントを初期化し、`google.generativeai.GenerativeModel` インスタンスを生成します。
  * **詳細**: UIでユーザーが選択した `model_name` を引数として受け取り、そのモデル名を使用してLLMをセットアップします。これにより、利用するGeminiモデルを動的に切り替えることが可能です。

### ReActループの実装 (`run_agent_turn`)

Gemini API の `function_call` 機能と Python の `while` ループを組み合わせて ReAct を実現しています。

```mermaid
flowchart TD
    User([ユーザー入力]) --> Start
    subgraph Agent_Process [run_agent_turn 関数]
        direction TB
        Start[開始] --> ReAct_Phase

        subgraph ReAct_Phase [Phase 1: ReAct Loopの実装]
            Think[思考 Thought] --> Decide{ツール必要?}
            Decide -- Yes --> Action[行動: part.function_call]
            Action --> Observe[観察: 検索結果取得]
            Observe --> Think
            Decide -- No --> Draft[ドラフト回答生成]
        end

        Draft --> Reflection_Phase

        subgraph Reflection_Phase [Phase 2: Reflectionの実装]
            Review[推敲プロンプト送信] --> Critique[自己評価 & 修正]
            Critique --> Finalize[最終回答抽出]
        end
    end
    Finalize --> Output([ユーザーへの回答])
```

* **コード対応**: `run_agent_turn` 関数内の `while turn_count < max_turns:` ループ。
* **Thoughtの可視化**: モデルが出力する `Thought:` パートを抽出し、Streamlit UI (`st.expander`) にリアルタイム表示します。

### プロンプト設計

* **Router Guidelines (`SYSTEM_INSTRUCTION_TEMPLATE`)**:
  * **役割**: どのコレクション（`wikipedia_ja`, `livedoor`, `cc_news`）を使うべきかの判断基準を提供します。
  * **実装**: LLMはこのガイドラインに従い、自律的に適切なコレクションを選択します。
* **Reflection Strategy (`REFLECTION_INSTRUCTION`)**:
  * **役割**: ドラフト回答に対する評価基準（正確性・適切性・スタイル）を定義します。

## 2.5 ツール定義: `agent_tools.py`

LLM が呼び出すことができる「手足」となる関数群です。

* **`search_rag_knowledge_base(query, collection_name)`**:
  * **役割**: 指定されたコレクションに対して検索を実行します。
  * **詳細**: `services.qdrant_service.search_collection_rag` をラップし、LLMが使いやすいインターフェースを提供します。
* **`list_rag_collections()`**:
  * **役割**: 現在利用可能なコレクションの一覧を返します。

## 2.6 知識ベース検索: `services/qdrant_service.py`

Qdrant データベースとの対話、Embedding 生成、ハイブリッド検索を担当するコアモジュールです。

### Embedding (ベクトル化) の構成

`helper_embedding.py` に集約され、抽象化されています。


| 項目               | 詳細                                                                                                                                                |
| :----------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| **抽象基底クラス** | `EmbeddingClient`                                                                                                                                   |
| **実装クラス**     | 1.**`GeminiEmbedding`**: Gemini API (`models.embed_content`) を使用。現在の主力。<br>2. **`OpenAIEmbedding`**: OpenAI API を使用。レガシー/互換用。 |
| **ファクトリ関数** | `create_embedding_client(provider="gemini", ...)`                                                                                                   |

### 検索ロジック (Hybrid Search)

Qdrant の **Hybrid RAG (Dense + Sparse)** 機能を活用しています。


| 処理フェーズ       | モジュール / 関数                                             | 詳細 (Input / Process / Output)                                                                                                                                                                 |
| :----------------- | :------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **設定 (Setup)**   | `qdrant_client_wrapper.py`<br>`create_or_recreate_collection` | **Input**: `client`, `name`, `vector_size`<br>**Process**: DenseベクトルとSparseベクトルの両方の設定を行い、コレクションを作成。<br>**Output**: なし                                            |
| **実行 (Runtime)** | `qdrant_client_wrapper.py`<br>`search_collection`             | **Input**: `client`, `collection_name`, `query_vector`<br>**Process**: Dense (意味検索) と Sparse (キーワード検索) を組み合わせたハイブリッド検索を実行。<br>**Output**: 高精度な検索結果リスト |

```mermaid
graph TD
    subgraph Python App
        Query[ユーザー入力] --> |helper_embedding.py| Embed[Embedding生成]
        Embed --> |GeminiEmbedding| API[Gemini API]
        API --> |Vector| Embed
        Embed --> |Vector| Search[search_collection]
    end

    subgraph Qdrant DB
        Search --> |Query Vector| Engine[検索エンジン]
        Config[Hybrid Search] -.-> Engine
        Engine --> |Dense + Sparse| Score[類似度スコア算出]
        Score --> |Top K Results| Search
    end
```

---

## 第3部: 動作シーケンス (Runtime Behavior)

## 3.1 処理シーケンス図

Streamlit UI、Gemini API、Agent Tools 間のインタラクション詳細です。

```mermaid
sequenceDiagram
    participant UI as Agent Controller<br>(Streamlit)
    participant LLM as 選択されたGeminiモデル
    participant Tool as Agent Tools<br>(Search/RAG)

    Note over UI, LLM: Phase 1: ReAct Loop
    UI->>LLM: send_message(ユーザー入力)
    loop 解決するまで繰り返し
        LLM-->>UI: 応答 (Text + FunctionCall?)
        alt Function Call あり
            UI->>UI: Thoughtをログ表示
            UI->>Tool: ツール実行 (例: search_rag)
            Tool-->>UI: 検索結果 (Observation)
            UI->>LLM: send_message(function_response)
        else Function Call なし
            LLM-->>UI: 回答案 (Draft Answer)
            Note over UI: ループ終了 (break)
        end
    end
```

#### 主要構成要素


| 項目                   | 実装詳細                         | 役割                                                                                     |
| :--------------------- | :------------------------------- | :--------------------------------------------------------------------------------------- |
| **ループ制御**         | `while turn_count < max_turns:`  | 思考・行動サイクルの維持と無限ループ防止。                                               |
| **ツール実行**         | `part.function_call` 検知        | モデルがツール利用を要求した場合、対応する Python 関数 (`agent_tools.py`) を実行します。 |
| **結果フィードバック** | `chat_session.send_message(...)` | ツールの実行結果を`function_response` としてモデルに返し、次の思考を促します。           |

## 3.2 Router & Multi-turn Strategy

エージェントがどのように検索対象を決定し、失敗時にリカバリするかを示します。

1. **Router (コレクション選択)**:
   * ユーザー入力の内容に基づき、`SYSTEM_INSTRUCTION` のルールに従って最適なコレクションを決定します（例: 一般知識なら `wikipedia_ja`）。
2. **Multi-turn Strategy (リカバリ)**:
   * 検索結果が `[[NO_RAG_RESULT]]` だった場合、LLM は即座に諦めず、**別のコレクション**を試したり、**クエリを言い換え**て再検索を行います。

---

# 第4部: モジュール構成図 (Module Dependencies)

システムの全体的な依存関係図です。

```mermaid
graph TD
    subgraph UI_Layer [UI / Controller]
        AgentPage[agent_chat_page.py<br>Main Logic]
    end

    subgraph Service_Layer [Services & Tools]
        Tools[agent_tools.py<br>Tool Definitions]
        QdrantSvc[services/qdrant_service.py<br>DB Access]
        Config[config.py]
    end

    subgraph External_API
        Gemini[google.generativeai]
        QdrantDB[(Qdrant Vector DB)]
    end

    AgentPage --> |Import/Call| Tools
    AgentPage --> |Import/Call| QdrantSvc
    AgentPage --> |Import| Config
    AgentPage --> |API Call| Gemini
    Tools --> |Search| QdrantSvc
    QdrantSvc --> |Query| QdrantDB
```

## =================================================

# RAG Q/A 生成・検索システム

本プロジェクトのメインアプリケーションは `agent_rag.py` です。
日本語・英語RAG（Retrieval-Augmented Generation）システム。ドキュメントからQ/Aペアを自動生成し、Qdrantベクトルデータベースで類似度検索・AI応答生成を行う統合アプリケーション。

![lp.png](assets/lp.png)

目次

- [1. 概要](#1-概要)
- [2. クイックスタート](#2-クイックスタート)
- [3. 統合アプリ rag_qa_pair_qdrant.py](#3-統合アプリ-rag_qa_pair_qdrantpy)
- [4. 技術コンポーネント](#4-技術コンポーネント)
  - [4.1 チャンク分割技術詳細（SemanticCoverage）](#41-チャンク分割技術詳細semanticcoverage)
  - [4.2 プロンプト設計・処理方式詳細](#42-プロンプト設計・処理方式詳細)
  - [4.3 Q/Aペア生成](#43-qaペア生成)
  - [4.4 Embedding・Qdrant登録](#44-embeddingqdrant登録)
  - [4.5 ベクトル検索・RAG](#45-ベクトル検索rag)
- [5. 環境構築詳細](#5-環境構築詳細)
- [6. プログラム一覧](#6-プログラム一覧)
- [7. ドキュメント一覧](#7-ドキュメント一覧)
- [8. ディレクトリ構造](#8-ディレクトリ構造)
- [9. 対応データセット](#9-対応データセット)
- [10. 技術スタック](#10-技術スタック)
- [11. ライセンス・貢献](#11-ライセンス貢献)

---

## 1. 概要

### 1.1 プロジェクト説明

本システムは、日本語、英語ドキュメントからQ/Aペアを自動生成し、ベクトル検索による質問応答（RAG）を実現する統合アプリケーションです。

### 1.2 主要機能


| 機能                | 説明                                                |
| ------------------- | --------------------------------------------------- |
| **チャンク分割**    | 段落優先・MeCab活用のセマンティック分割で意味を保持 |
| **プロンプト設計**  | 2段階構成・構造化出力による高品質なQ/A生成制御      |
| **Q/Aペア自動生成** | LLM（GPT-4o等）を使用してドキュメントからQ/Aを生成  |
| **Celery並列処理**  | 大規模データの高速処理（24ワーカーで約20倍高速化）  |
| **Qdrant登録**      | Q/AペアをEmbedding化してベクトルDBに登録            |
| **類似度検索**      | コサイン類似度によるセマンティック検索              |
| **RAG応答生成**     | 検索結果を基にAIが回答を生成                        |
| **カバレージ分析**  | Q/Aがドキュメントをどの程度網羅しているか評価       |

### 1.3 システムアーキテクチャ

```mermaid
flowchart TB
        APP["agent_rag.py 統合アプリ"]
        A1["説明<br/>About"]
        A2["RAGデータDL"]
        A3["Q/A生成"]
        A4["Qdrant登録"]
        A5["Show Qdrant"]
        A6["Qdrant検索"]
    end

    subgraph PIPELINE["処理パイプライン"]
        P1["データセット<br/>cc_news / livedoor / wikipedia"]
        P2["チャンク分割<br/>SemanticCoverage<br/>段落優先 / 文分割 / MeCab"]
        P3["Q/A生成<br/>LLM / Celery<br/>GPT-4o-mini / 並列処理"]
        P4["Embedding<br/>OpenAI API<br/>text-embedding-3-small<br/>1536次元"]
        P5["Qdrant登録<br/>Vector DB<br/>コサイン類似度検索"]
    end

    A1 --> PIPELINE
    A2 --> PIPELINE
    A3 --> PIPELINE
    A4 --> PIPELINE
    A5 --> PIPELINE
    A6 --> PIPELINE

    P1 --> P2 --> P3 --> P4 --> P5
```

### 1.4 処理パイプライン（データフロー）

```mermaid
flowchart LR
    subgraph INPUT["入力"]
        I1["データセット<br/>cc_news / livedoor等"]
        I2["ユーザー質問"]
    end

    subgraph PROCESS["処理"]
        P1["1. チャンク分割<br/>SemanticCoverage"]
        P2["2. Q/A生成<br/>LLM + Celery"]
        P3["3. Embedding<br/>OpenAI API"]
        P4["4. Qdrant登録<br/>PointStruct"]
        P5["5. 検索 + RAG<br/>類似度検索"]
    end

    subgraph OUTPUT["出力"]
        O1["chunks<br/>段落/文単位"]
        O2["qa_pairs<br/>question / answer"]
        O3["vectors<br/>1536次元"]
        O4["Collection<br/>id / vector / payload"]
        O5["AI応答"]
    end

    I1 --> P1 --> O1
    P1 --> P2 --> O2
    P2 --> P3 --> O3
    P3 --> P4 --> O4
    I2 --> P5 --> O5
    P4 --> P5
```

---

## 2. クイックスタート

### 2.1 前提条件

- Python 3.10以上
- Docker / Docker Compose
- OpenAI APIキー

### 2.2 インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd openai_rag_qa_jp

# 依存パッケージのインストール
pip install -r requirements.txt

# 環境変数の設定
cp .env.example .env
# .envにOPENAI_API_KEYを設定
```

### 2.3 サービス起動

```bash
# Qdrant + Redis の起動
docker-compose -f docker-compose/docker-compose.yml up -d

# Celeryワーカー起動（並列処理を使う場合）
./start_celery.sh start -w 8

# 統合アプリの起動
streamlit run agent_rag.py
```

### 2.4 動作確認

ブラウザで http://localhost:8501 を開き、統合アプリが表示されることを確認。

**詳細な環境構築手順**: [doc/01_install.md](doc/01_install.md)

---

### 3. 統合アプリ agent_rag.py

### 3.1 6画面構成

統合アプリは以下の6つの画面で構成されています。


| # | 画面名          | 機能             | 主な操作                          |
| - | --------------- | ---------------- | --------------------------------- |
| 1 | **説明**        | プロジェクト概要 | ドキュメント確認                  |
| 2 | **RAGデータDL** | データセット取得 | cc_news, livedoor等のダウンロード |
| 3 | **Q/A生成**     | Q/Aペア生成      | LLM生成、Celery並列処理           |
| 4 | **Qdrant登録**  | ベクトルDB登録   | CSV→Embedding→登録              |
| 5 | **Show-Qdrant** | コレクション表示 | データ確認、統計情報              |
| 6 | **Qdrant検索**  | 類似度検索       | 質問入力→検索→AI応答            |

![RAG_dl.png](assets/RAG_dl.png?t=1764367055620)

### 3.2 画面フロー

```mermaid
flowchart LR
    S1["説明"] --> S2["RAGデータDL"] --> S3["Q/A生成"] --> S4["Qdrant登録"] --> S6["Qdrant検索"]
    S3 --> S5["Show-Qdrant<br/>データ確認"]
```

### 3.3 各画面の概要

![RAG_dl.png](assets/RAG_dl.png?t=1764367120205)

#### 画面1: 説明（About）

プロジェクトの概要とドキュメントへのリンクを表示。

#### 画面2: RAGデータDL

Hugging Faceからデータセットをダウンロード・前処理。

- 対応データセット: cc_news, livedoor, wikipedia_ja

#### 画面3: Q/A生成

チャンク分割 → LLMによるQ/Aペア生成。

- 同期処理 / Celery並列処理を選択可能
- カバレージ分析オプション

#### 画面4: Qdrant登録

CSVファイルからQdrantへベクトルデータを登録。

- Embedding生成（text-embedding-3-small）
- コレクション統合機能

#### 画面5: Show-Qdrant

登録済みコレクションの確認・統計表示。

#### 画面6: Qdrant検索

質問を入力 → 類似Q/A検索 → AI応答生成。

**詳細な操作方法**: [doc/02_rag.md](doc/02_rag.md)

---

## 4. 技術コンポーネント

### 4.1 チャンク分割技術詳細（SemanticCoverage）

本システムでは、`SemanticCoverage` クラス（`helper_rag_qa.py`）を用いて、文書を意味的に一貫性のあるチャンクに分割します。

#### 4.1.1 システム概要と役割

チャンク分割は以下の重要な役割を担います：

1. **適切なサイズへの分割**: Embeddingモデルの最適入力長（デフォルト300トークン）に収める。
2. **意味的一貫性の維持**: 段落や文の境界を尊重し、文脈の断絶を防ぐ。
3. **トピック連続性の確保**: 短すぎるチャンクを統合し、情報の断片化を防ぐ。

**システム構成とフロー:**

```mermaid
flowchart TD
    A["create_semantic_chunks()"] --> B{"prefer_paragraphs?"}
    B -->|"True"| C["段落分割 (_split_into_paragraphs)"]
    C --> D{"段落 <= max_tokens?"}
    D -->|"Yes"| E["paragraphチャンク"]
    D -->|"No"| F["文分割 (_split_into_sentences)"]
    F --> G["sentence_groupチャンク"]
    G --> H["トピック連続性調整 (_adjust_chunks...)"]
    E --> H
    H --> I["最終チャンクリスト"]
```

#### 4.1.2 SemanticCoverageクラス詳細

- **クラス**: `SemanticCoverage`
- **主要メソッド**:
  - `create_semantic_chunks()`: メイン分割処理。段落優先フラグ(`prefer_paragraphs`)により挙動を制御。
  - `_split_into_paragraphs()`: 空行（`\n\n`）を基準に、筆者が意図した意味的まとまりを抽出。
  - `_split_into_sentences()`: 日本語（MeCab/正規表現）および英語の文分割。
  - `_adjust_chunks_for_topic_continuity()`: `min_tokens`（デフォルト50）未満のチャンクを前後のチャンクとマージ。

#### 4.1.3 分割ロジックと実装技術

1. **段落優先アプローチ**:
   正規表現 `r'\n\s*\n'` を用いて、文書の構造的な段落を最優先の境界として扱います。これにより、トピックの急激な切り替わりを防ぎます。
2. **MeCabによる高精度な文分割**:
   日本語テキストの場合、自動判定ロジックにより `MeCab` を優先的に使用します。

   - **利点**: 「3.14」のような数値内のピリオドや、括弧内の句点などを正しく認識し、正規表現単独で発生する誤分割を防止します。
   - **フォールバック**: MeCabが利用できない環境では、自動的に正規表現ベースの分割（`(?<=[。．.!?])\s*`）に切り替わります。
3. **強制分割（Forced Split）**:
   単一の文が `max_tokens` を超える稀なケース（条文やログデータなど）では、トークン数に基づいて強制的に分割し、モデルの入力制限超過を防ぎます。

**詳細**: [doc/03_chunk.md](doc/03_chunk.md)

### 4.2 プロンプト設計・処理方式詳細

Geminiモデル（`gemini-2.0-flash`等）の性能を最大限に引き出すため、`UnifiedLLMClient` を介した2段階のプロンプト構成と構造化出力を採用しています。

#### 4.2.1 プロンプト設計の特徴

- **2段階プロンプト構造**:
  - **システムプロンプト**: AIの役割（「教育コンテンツ作成の専門家」）と全般的な生成ルールを定義。
  - **ユーザープロンプト**: 処理対象のテキストチャンク、生成数指示、JSONスキーマを含みます。
- **言語別テンプレート**: 日本語(`ja`)と英語(`en`)で最適化されたプロンプトを自動切り替え。
- **動的調整**: チャンクのトークン数や複雑度に応じて、生成するQ/Aペア数（2〜5ペア）を動的に決定。

#### 4.2.2 システムプロンプト定義（日本語版）

```text
あなたは教育コンテンツ作成の専門家です。
与えられた日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成
```

#### 4.2.3 ユーザープロンプトと構造化出力

ユーザープロンプトでは、明確なJSON形式での出力を指示し、以下の質問タイプを網羅するように誘導します。

- **質問タイプ階層**:
  - `fact`: 事実確認型
  - `reason`: 理由説明型
  - `comparison`: 比較型
  - `application`: 応用型

**UnifiedLLMClientによる型安全な実装**:
Gemini APIの `response_schema` (またはOpenAIの Structured Outputs) を利用し、Pydanticモデル (`QAPairsResponse`) に準拠したデータを強制的に生成させます。

```python
# helper_llm.py 呼び出しイメージ
response: QAPairsResponse = client.generate_structured(
    prompt=combined_prompt,
    response_schema=QAPairsResponse, # Pydanticモデル
    model="gemini-2.0-flash"
)
```

**詳細**: [doc/04_prompt.md](doc/04_prompt.md)

### 4.3 Q/Aペア生成

Celery並列処理によるスケーラブルなQ/A生成。

```mermaid
flowchart LR
    subgraph SYNC["同期処理"]
        S1["1チャンク"] --> S2["1API呼び出し"]
        S2 --> S3["約50分/1000チャンク"]
    end

    subgraph ASYNC["非同期処理 Celery"]
        A1["N個のワーカー"] --> A2["並列実行"]
        A2 --> A3["約2-3分/1000チャンク<br/>24ワーカー"]
    end
```

**主要パラメータ:**


| パラメータ         | デフォルト | 説明                     |
| ------------------ | ---------- | ------------------------ |
| --use-celery       | false      | 並列処理を有効化         |
| --celery-workers   | 4          | ワーカー数               |
| --batch-chunks     | 3          | 1API呼び出しのチャンク数 |
| --analyze-coverage | false      | カバレージ分析           |

**詳細**: [doc/05_qa_pair.md](doc/05_qa_pair.md)

### 4.4 Embedding・Qdrant登録

Q/AペアをベクトルDBに登録するフロー。


| 項目            | 設定                   |
| --------------- | ---------------------- |
| Embeddingモデル | text-embedding-3-small |
| ベクトル次元    | 1536                   |
| 距離メトリクス  | コサイン類似度         |
| バッチサイズ    | 128ポイント/バッチ     |

**コレクション命名規則:**

```
qa_{dataset}_{method}
例: qa_cc_news_a02_llm, qa_livedoor_a03_rule
```

**詳細**: [doc/06_embedding_qdrant.md](doc/06_embedding_qdrant.md)

### 4.5 ベクトル検索・RAG

類似度検索とAI応答生成。

```mermaid
flowchart TB
    Q["ユーザー質問"]
    E["Embedding生成"]
    S["Qdrant検索"]
    T["Top-K Q/A取得"]
    R["RAG応答生成"]
    A["最終回答"]

    Q --> E
    E --> S
    S --> T
    T --> R
    R --> A

    E -.- E1["OpenAI API"]
    S -.- S1["HNSW近似最近傍探索"]
    R -.- R1["GPT-4o-mini"]
```

**スコア解釈:**


| スコア     | 解釈             | 推奨アクション         |
| ---------- | ---------------- | ---------------------- |
| 0.90〜1.00 | 極めて高い類似度 | そのまま回答として使用 |
| 0.80〜0.89 | 高い類似度       | AI回答で補完推奨       |
| 0.70〜0.79 | 中程度           | 参考情報として表示     |
| 0.60未満   | 低い類似度       | 別の検索を促す         |

**詳細**: [doc/07_qdrant_integration_add.md](doc/07_qdrant_integration_add.md)

---

## 5. 環境構築詳細

### 5.1 Python環境

```bash
# Python 3.10以上が必要
python --version

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

### 5.2 依存パッケージ

```bash
pip install -r requirements.txt

# Celery関連（並列処理を使う場合）
pip install "celery[redis]" kombu flower
```

### 5.3 Docker（Qdrant + Redis）

```bash
# docker-compose.ymlの場所
docker-compose -f docker-compose/docker-compose.yml up -d

# 起動確認
curl http://localhost:6333/collections  # Qdrant
redis-cli ping                           # Redis
```

### 5.4 MeCab（日本語形態素解析）

```bash
# Mac
brew install mecab mecab-ipadic

# Ubuntu
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8

# Python バインディング
pip install mecab-python3
```

### 5.5 環境変数

`.env`ファイルを作成:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
```

**詳細な環境構築手順**: [doc/01_install.md](doc/01_install.md)

---

## 6. プログラム一覧

### 6.1 統合アプリ


| ファイル       | 説明                           |
| -------------- | ------------------------------ |
| `agent_rag.py` | 6画面構成の統合Streamlitアプリ |
| `server.py`    | Qdrantサーバー管理スクリプト   |

### 6.2 データ処理・Q&A生成


| ファイル                   | 説明                         |
| -------------------------- | ---------------------------- |
| `a01_load_set_rag_data.py` | データセットのロード・前処理 |
| `a02_make_qa_para.py`      | Q/A生成（LLM + Celery並列）  |
| `a03_make_qa_rule.py`      | Q/A生成（ルールベース）      |
| `a10_make_qa_hybrid.py`    | Q/A生成（ハイブリッド）      |
| `celery_tasks.py`          | Celeryタスク定義             |

### 6.3 ベクトルストア・検索


| ファイル                         | 説明                        |
| -------------------------------- | --------------------------- |
| `a30_qdrant_registration.py`     | Qdrantへのデータ登録（CLI） |
| `a35_qdrant_truncate.py`         | Qdrantコレクション削除      |
| `a40_show_qdrant_data.py`        | Qdrantデータ表示            |
| `a50_rag_search_local_qdrant.py` | RAG検索（CLI/Streamlit）    |

### 6.4 サービス層


| ファイル                     | 説明                        |
| ---------------------------- | --------------------------- |
| `services/qdrant_service.py` | Qdrant操作サービス          |
| `services/qa_service.py`     | Q/A生成サービス             |
| `helper_api.py`              | OpenAI API ユーティリティ   |
| `helper_rag.py`              | RAGデータ処理ユーティリティ |
| `rag_qa.py`                  | SemanticCoverageクラス      |

---

## 7. ドキュメント一覧

### 7.1 ドキュメント相関図

```mermaid
flowchart TB
    README["README.md<br/>プロジェクト概要"]

    README --> D1["doc/01_install.md<br/>環境構築"]
    README --> D2["doc/02_rag.md<br/>統合アプリ操作"]
    README --> D3["doc/03-07<br/>技術詳細"]

    D2 --> D3_1["03_chunk.md<br/>チャンク"]
    D2 --> D3_2["04_prompt.md<br/>プロンプト"]
    D2 --> D3_3["05_qa_pair.md<br/>Q/A生成"]
    D2 --> D3_4["06_embedding_qdrant.md<br/>Embedding"]
    D2 --> D3_5["07_qdrant_integration_add.md<br/>検索・統合"]
```

### 7.2 ドキュメント概要


| ドキュメント                                                         | 主題                     | 対象読者       |
| -------------------------------------------------------------------- | ------------------------ | -------------- |
| [doc/01_install.md](doc/01_install.md)                               | 環境構築ガイド           | 導入者・開発者 |
| [doc/02_rag.md](doc/02_rag.md)                                       | 統合アプリ操作マニュアル | 利用者・開発者 |
| [doc/03_chunk.md](doc/03_chunk.md)                                   | チャンク分割技術         | 開発者         |
| [doc/04_prompt.md](doc/04_prompt.md)                                 | プロンプト設計           | 開発者         |
| [doc/05_qa_pair.md](doc/05_qa_pair.md)                               | Q/Aペア生成処理          | 開発者         |
| [doc/06_embedding_qdrant.md](doc/06_embedding_qdrant.md)             | Embedding・Qdrant登録    | 開発者         |
| [doc/07_qdrant_integration_add.md](doc/07_qdrant_integration_add.md) | Qdrant検索・統合         | 開発者         |

---

## 8. ディレクトリ構造

```
openai_rag_qa_jp/
├── agent_rag.py      # 統合アプリ（メインエントリポイント）
├── server.py                   # サーバー管理
│
├── a01_load_set_rag_data.py   # データロード
├── a02_make_qa_para.py        # Q/A生成（LLM）
├── a03_make_qa_rule.py        # Q/A生成（ルール）
├── a10_make_qa_hybrid.py      # Q/A生成（ハイブリッド）
├── a30_qdrant_registration.py # Qdrant登録
├── a50_rag_search_local_qdrant.py # RAG検索
│
├── celery_tasks.py            # Celeryタスク
├── config.py                  # 設定管理
├── models.py                  # Pydanticモデル
│
├── helper_api.py              # OpenAI API ユーティリティ
├── helper_rag.py              # RAGデータ処理
├── rag_qa.py                  # SemanticCoverageクラス
│
├── services/                  # サービス層
│   ├── qdrant_service.py      # Qdrant操作
│   └── qa_service.py          # Q/A生成
│
├── ui/                        # UIコンポーネント
│   └── pages/                 # 各画面のページ
│       ├── qa_generation_page.py
│       ├── qdrant_registration_page.py
│       ├── qdrant_search_page.py
│       └── qdrant_show_page.py
│
├── doc/                       # ドキュメント
│   ├── 01_install.md
│   ├── 02_rag.md
│   ├── 03_chunk.md
│   ├── 04_prompt.md
│   ├── 05_qa_pair.md
│   ├── 06_embedding_qdrant.md
│   └── 07_qdrant_integration_add.md
│
├── docker-compose/            # Docker設定
│   └── docker-compose.yml
│
├── qa_output/                 # 生成されたQ/Aデータ
├── OUTPUT/                    # 前処理済みデータ
│
├── requirements.txt           # 依存パッケージ
├── .env                       # 環境変数（gitignore）
├── config.yml                 # アプリ設定
└── CLAUDE.md                  # Claude Code用ガイド
```

---

## 9. 対応データセット


| データセット  | 言語   | 内容          | ソース       |
| ------------- | ------ | ------------- | ------------ |
| cc_news       | 日本語 | ニュース記事  | Hugging Face |
| livedoor      | 日本語 | ブログ記事    | Hugging Face |
| wikipedia_ja  | 日本語 | Wikipedia記事 | Hugging Face |
| japanese_text | 日本語 | 汎用テキスト  | カスタム     |

---

## 10. 技術スタック


| カテゴリ       | 技術                          |
| -------------- | ----------------------------- |
| **言語**       | Python 3.10+                  |
| **LLM**        | OpenAI GPT-4o, GPT-4o-mini    |
| **Embedding**  | OpenAI text-embedding-3-small |
| **ベクトルDB** | Qdrant                        |
| **並列処理**   | Celery + Redis                |
| **Web UI**     | Streamlit                     |
| **形態素解析** | MeCab                         |
| **コンテナ**   | Docker / Docker Compose       |

---

## 11. ライセンス・貢献

### ライセンス

MIT License

### 貢献

1. Issueを作成して問題を報告
2. Pull Requestで改善を提案
3. ドキュメントの改善も歓迎

### フィードバック

問題報告・機能要望は [GitHub Issues](https://github.com/anthropics/claude-code/issues) へ

---

**最終更新**: 2025-11-29
