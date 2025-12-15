#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
agent_chat_page.py - ハイブリッド・ナレッジ・エージェント チャット画面
================================================================
Gemini 2.0 Flash を使用した ReAct 型エージェントとの対話インターフェース。
Qdrant 上のナレッジベース（コレクション）を動的に選択し、RAG 検索を行いながら回答します。
"""

import os
import logging
import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.generativeai import ChatSession, GenerativeModel
from typing import Dict, List, Any, Optional, Union, Tuple
from qdrant_client import QdrantClient

# 設定とツール
from config import AgentConfig, GeminiConfig
from agent_tools import search_rag_knowledge_base, list_rag_collections, RAGToolError
from services.qdrant_service import get_all_collections
from services.log_service import log_unanswered_question

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 定数・設定
# -----------------------------------------------------------------------------

SYSTEM_INSTRUCTION_TEMPLATE = """
あなたは、社内ドキュメント検索システムと連携した「ハイブリッド・ナレッジ・エージェント」です。
あなたの役割は、ユーザーの質問に対して、一般的な知識と、提供されたツール（社内ナレッジ検索）を適切に使い分けて回答することです。

## ReAct プロセスと出力フォーマット (厳守)

あなたは **Thought (思考)**、**Action (ツール実行)**、**Observation (結果観察)** のサイクルを回して回答に到達する必要があります。

### 1. ツールを使用する場合（検索が必要な場合）
必ず以下の形式で思考を出力してから、ツールを呼び出してください。
**Thought: [なぜ検索が必要か、どのコレクションを、どんなクエリで検索するか]**
(この後にツール呼び出しが行われます)

### 2. 最終回答を行う場合（検索が完了した、または検索不要な場合）
必ず以下の形式で出力してください。
**Thought: [得られた情報に基づいてどう回答するか、または検索結果がなかった場合の判断]**
**Answer: [ユーザーへの最終的な回答]**

---

## 行動指針 (Router Guidelines)

1.  **専門知識の検索**:
    *   以下のいずれかに該当する場合は、**必ず `search_rag_knowledge_base` ツールを使用してください。**
        *   プロジェクト固有の仕様、設定、エラー、社内規定、Wikipediaの知識に関する質問。
        *   特定の情報源（例: "Wikipediaによると"、"ライブドアニュースで"）が指定されている質問。
        *   **内容が不明瞭であっても、社内ナレッジに関連する可能性があると判断される質問（例：特定のコード名、システム名、ランダムに見える文字列など）。**
        *   **ただし、一般的なプログラミング言語の文法や使い方に関する質問にはツールを使用しないでください。**
    *   **現在利用可能なコレクションは以下の通りです:**
        {available_collections}

2.  **コレクション選択のヒント (言語と内容のマッチング)**:
    *   質問の言語と内容に応じて、最適なコレクションを選択してください。
    *   **`cc_news`**: **英語 (English)** のニュース記事。 **英語の質問にはまずこれを使用してください。検索クエリも英語のままにしてください。**
    *   **`wikipedia_ja`**: 日本語 (Japanese) の百科事典。一般的な知識や定義。
    *   **`livedoor`**: 日本語 (Japanese) のニュース・ブログ。**日本のニュース、エンタメ、映画などの話題にはまずこれを使用してください。**
    *   **`japanese_text`**: 日本語 (Japanese) のWebテキスト。**他の日本語コレクションで結果が出ない場合の予備として使用してください。**

3.  **再試行戦略 (Multi-turn Strategy)**:
    *   **Step 1 (初回検索):** 質問内容に最も適したコレクションを選びます。(英語なら `cc_news`、日本のニュース・エンタメなら `livedoor`、一般知識なら `wikipedia`)
    *   **Step 2 (結果の評価):** もし検索結果が `[[NO_RAG_RESULT]]` (結果なし) だった場合、**すぐに諦めずに以下の戦略をとってください。**
        *   **コレクション変更:** 別のコレクションを試してください。例えば `livedoor` で見つからなければ `wikipedia_ja` を、それでもなければ `japanese_text` を検索してください。
        *   **クエリ変更:** キーワードを少し広げる、または同義語に変えて再検索する。英語コレクションには英語で、日本語コレクションには日本語で検索するよう注意してください。
    *   **Step 3 (諦め):** 複数のコレクションを試行しても情報が見つからない場合のみ、「情報が見つかりませんでした」と回答してください。

4.  **一般的な会話**:
    *   挨拶、雑談、単純な計算など、専門知識が不要な場合は、ツールを使わずに `Answer:` で直接回答してください。

5.  **正直さと不足情報の処理 (Critical)**:
    *   ツール検索の結果、情報が得られなかった場合は、**絶対に**あなたの事前学習知識で捏造してはいけません。
    *   「提供された社内ナレッジには関連情報がありませんでした」と正直に伝えてください。

6.  **回答のスタイル**:
    *   丁寧な日本語（です・ます調）で回答してください。
    *   検索結果に基づく回答の場合、「社内ナレッジによると...」や「ソース [ファイル名] によると...」と出典を明示してください。
"""

REFLECTION_INSTRUCTION = """
## Reflection (自己評価と修正)

あなたは上記で作成した「回答案」を、以下の基準で客観的に評価し、必要であれば修正してください。

**チェックリスト:**
1.  **正確性:** 検索結果(もしあれば)に基づいているか？ 提供された情報源に含まれない情報を捏造していないか？
2.  **回答の適切性:** ユーザーの質問に直接的かつ明確に答えているか？
3.  **スタイル:** 親しみやすく、丁寧な日本語（です・ます調）か？ 箇条書きなどを活用して読みやすいか？

**指示:**
*   修正が不要な場合でも、必ず **Final Answer** を出力してください。
*   修正が必要な場合は、修正後の回答を **Final Answer** として出力してください。
*   思考プロセスは `Thought:` で始めてください。

**出力フォーマット:**
Thought: [評価と修正の思考プロセス]
Final Answer: [最終的な回答]
"""

# ツールのマッピング
TOOLS_MAP = {
    'search_rag_knowledge_base': search_rag_knowledge_base,
    'list_rag_collections': list_rag_collections
}

# -----------------------------------------------------------------------------
# ヘルパー関数
# -----------------------------------------------------------------------------

def get_available_collections_from_qdrant() -> List[str]:
    """Qdrantから利用可能なコレクション名を取得"""
    try:
        # qdrant_service.py のロジックを利用しても良いが、シンプルにクライアントから取得
        client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        collections = client.get_collections()
        return [c.name for c in collections.collections]
    except Exception as e:
        logger.error(f"Failed to fetch collections: {e}")
        return []

def setup_agent(selected_collections: List[str], model_name: str) -> ChatSession:
    """Geminiエージェントのセットアップ"""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY or GOOGLE_API_KEY not set.")
        raise ValueError("API Key missing")
    
    genai.configure(api_key=api_key)
    
    # 利用可能なツール
    tools_list = [search_rag_knowledge_base, list_rag_collections]
    
    # システムプロンプトに利用可能なコレクションを埋め込む
    collections_str = ", ".join(selected_collections) if selected_collections else "(コレクションが見つかりません)"
    system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(available_collections=collections_str)
    
    model = genai.GenerativeModel(
        model_name=model_name,
        tools=tools_list,
        system_instruction=system_instruction
    )
    
    chat = model.start_chat(enable_automatic_function_calling=False)
    return chat

def run_agent_turn(chat_session: ChatSession, user_input: str) -> str:
    """
    エージェントの1ターンを実行（ReActループ）
    UI向けに、思考プロセスやツール実行結果を逐次表示するよう改良。
    """
    
    response = chat_session.send_message(user_input)
    final_response_text = ""
    
    # ループ回数制限（無限ループ防止）
    max_turns = 10
    turn_count = 0
    
    thought_log = [] # 思考プロセスのログ
    
    while turn_count < max_turns:
        turn_count += 1
        function_call_found = False
        current_turn_text_from_model = "" # 現在のターンでモデルが生成したテキストを一時的に保持
        
        for part in response.parts:
            if part.text:
                text = part.text.strip()
                if "Thought:" in text or "考え:" in text:
                    thought_log.append(f"🧠 **Thought:**\n{text}")
                    logger.info(f"Agent Thought: {text}")
                    current_turn_text_from_model = text
                else:
                    # ツール呼び出しがなければこれが最終回答となる
                    current_turn_text_from_model = text
                    logger.info(f"Agent Response: {text}")

            if part.function_call:
                function_call_found = True
                fn = part.function_call
                tool_name = fn.name
                tool_args = dict(fn.args)
                
                logger.info(f"Agent Tool Call: {tool_name}({tool_args})")
                thought_log.append(f"🛠️ **Tool Call:** `{tool_name}`\nArgs: `{tool_args}`")
                
                # ツール実行
                tool_result = ""
                try:
                    if tool_name in TOOLS_MAP:
                        # ステータス表示
                        with st.spinner(f"ツールを実行中: {tool_name}..."):
                            tool_result = TOOLS_MAP[tool_name](**tool_args)
                    else:
                        tool_result = f"Error: Tool '{tool_name}' not found."
                except Exception as e:
                    tool_result = f"Error: {str(e)}"
                
                log_tool_result = str(tool_result)[:500] + "..." if len(str(tool_result)) > 500 else str(tool_result)
                thought_log.append(f"📝 **Tool Result:**\n{log_tool_result}")
                logger.info(f"Tool Result: {log_tool_result}")
                
                # 検索失敗（結果なし/低スコア）のログ記録
                if isinstance(tool_result, str) and tool_result.startswith("[[NO_RAG_RESULT"):
                    reason = "NO_RESULT"
                    if "LOW_SCORE" in tool_result:
                        reason = "LOW_SCORE"
                    
                    collection_arg = tool_args.get('collection_name', 'unknown')
                    log_unanswered_question(
                        query=user_input,
                        collections=[collection_arg],
                        reason=reason,
                        agent_response="(Search Failed)"
                    )

                # 結果をモデルに返す
                response = chat_session.send_message(
                    [genai.protos.Part(
                        function_response={
                            "name": tool_name,
                            "response": {'result': tool_result}
                        }
                    )]
                )
                break # response.parts のループを抜けて、次のモデル応答を処理
        
        if not function_call_found:
            # ツール呼び出しがなかった場合、現在のモデルのテキストが最終回答案(Draft)となる
            final_response_text = current_turn_text_from_model
            break
            
    # -------------------------------------------------------------------------
    # Phase 2: Reflection (自己洗練)
    # ReActで生成された回答案(final_response_text)を評価・修正する
    # -------------------------------------------------------------------------
    if final_response_text:
        with st.spinner("回答を推敲中 (Reflection)..."):
            try:
                # 思考ログへの区切り線

                thought_log.append("🔄 **Reflection Phase (推敲)**")

                # Reflectionプロンプトの送信
                reflection_msg = f"{REFLECTION_INSTRUCTION}\n\n**あなたの回答案:**\n{final_response_text}"
                reflection_response = chat_session.send_message(reflection_msg)
                
                reflection_text = reflection_response.text.strip()
                
                # 思考と回答の分離
                reflection_thought = ""
                reflection_answer = ""

                if "Final Answer:" in reflection_text:
                    parts = reflection_text.split("Final Answer:", 1)
                    reflection_thought = parts[0].strip()
                    reflection_answer = parts[1].strip()
                else:
                    # フォーマット崩れの場合はそのまま採用
                    reflection_thought = "Format mismatch in reflection."
                    reflection_answer = reflection_text

                # ログに追加
                if reflection_thought:
                    # Thought: タグがあれば除去して綺麗にする
                    clean_thought = reflection_thought.replace("Thought:", "").strip()
                    thought_log.append(f"🤔 **Reflection Thought:**\n{clean_thought}")
                    logger.info(f"Reflection Thought: {clean_thought}")

                if reflection_answer:
                    # 最終回答を更新
                    final_response_text = reflection_answer
                    logger.info(f"Reflection Answer: {reflection_answer}")

            except Exception as e:
                logger.error(f"Error during reflection phase: {e}")
                thought_log.append(f"⚠️ **Reflection Error:** {str(e)}")
                # エラー時はDraftをそのまま使う

    # 思考プロセスをexpanderで表示
    if thought_log:
        with st.expander("🤔 エージェントの思考プロセス (Click to open)", expanded=False):
            for i, log in enumerate(thought_log):
                st.markdown(log)
                if i < len(thought_log) - 1:
                    st.divider()

    # 最終回答の整形: Answer: タグがあればそこを抽出、なければ Thought: を除去
    if "Answer:" in final_response_text:
        # "Thought: ... Answer: ..." の形式から Answer 以降を取得
        parts = final_response_text.split("Answer:", 1)
        final_response_text = parts[1].strip()
    elif final_response_text.startswith("Thought:"):
        final_response_text = final_response_text.replace("Thought:", "").strip()
    elif final_response_text.startswith("考え:"):
        final_response_text = final_response_text.replace("考え:", "").strip()

    return final_response_text

# -----------------------------------------------------------------------------
# メイン画面表示関数
# -----------------------------------------------------------------------------

def show_agent_chat_page():
    st.title("🤖 エージェント対話 (Agent Chat)")
    st.caption("Gemini 2.0 Flash + ReAct + Qdrant Hybrid RAG (Dense + Sparse)")

    # -------------------------------------------------------------------------
    # 元ドキュメント表示エリア (Added)
    # -------------------------------------------------------------------------
    with st.expander("📄 元ドキュメントの表示", expanded=False):
        st.markdown("ダウンロードしたドキュメントの選択：")
        
        output_dir = "OUTPUT"
        target_patterns = {
            "cc_news": "cc_news*.txt",
            "japanese_text": "japanese_text*.txt",
            "livedoor": "livedoor*.txt",
            "wikipedia_ja": "wikipedia_ja*.txt"
        }
        
        file_options = {}
        if os.path.exists(output_dir):
            import glob
            for label, pattern in target_patterns.items():
                files = glob.glob(os.path.join(output_dir, pattern))
                if files:
                    # 更新日時順にソートして最新を取得
                    latest_file = max(files, key=os.path.getctime)
                    file_options[label] = latest_file
        
        if file_options:
            selected_doc_label = st.selectbox(
                "ドキュメントを選択:", 
                options=list(file_options.keys()),
                key="original_doc_selector"
            )
            
            if selected_doc_label:
                file_path = file_options[selected_doc_label]
                st.caption(f"参照ファイル: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = []
                        for _ in range(100):
                            line = f.readline()
                            if not line: break
                            lines.append(line)
                        st.text_area("ファイル内容 (先頭100行):", value="".join(lines), height=300)
                except Exception as e:
                    st.error(f"読み込みエラー: {e}")
        else:
            st.info("OUTPUTディレクトリにテキストファイルが見つかりません。")

    # -------------------------------------------------------------------------
    # 入力クエリの参考用 Q&A表示エリア (Added)
    # -------------------------------------------------------------------------
    with st.expander("📚 登録済みQ&Aの参照 (生成AI：Geminiが元ドキュメントの意味を解析しドキュメント内の重要箇所に基づいて「質問」と「回答」のペアを自動抽出しRAGシステムで利用可能なCSV形式のナレッジデータとして生成）入力クエリのヒント", expanded=False):
        st.markdown("登録されているコレクションから、質問と回答のサンプルを100件表示します。質問の参考にしてください。")
        
        # プレビュー用のコレクション取得
        preview_collections = get_available_collections_from_qdrant()
        
        if preview_collections:
            col1, col2 = st.columns([1, 3])
            with col1:
                target_collection = st.selectbox(
                    "コレクションを選択:", 
                    preview_collections,
                    index=0,
                    key="preview_collection_selector"
                )
            
            if target_collection:
                try:
                    # Qdrantクライアント接続
                    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
                    
                    # 上位100件を取得
                    points, _ = client.scroll(
                        collection_name=target_collection,
                        limit=100,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    if points:
                        data_list = []
                        for point in points:
                            payload = point.payload or {}
                            data_list.append({
                                "Question": payload.get("question", "N/A"),
                                "Answer": payload.get("answer", "N/A")
                            })
                        
                        df_preview = pd.DataFrame(data_list)
                        st.dataframe(
                            df_preview, 
                            use_container_width=True, 
                            hide_index=True,
                            column_config={
                                "Question": st.column_config.TextColumn("質問 (Question)", width="medium"),
                                "Answer": st.column_config.TextColumn("回答 (Answer)", width="large"),
                            }
                        )
                    else:
                        st.info(f"コレクション '{target_collection}' にデータが見つかりませんでした。")
                        
                except Exception as e:
                    st.error(f"データ取得エラー: {e}")
        else:
            st.warning("表示可能なコレクションがありません。Qdrantの状態を確認してください。")

    # 1. サイドバー設定
    with st.sidebar:
        st.header("⚙️ エージェント設定")
        
        # モデル選択の追加
        selected_model = st.selectbox(
            "使用モデル (Model)",
            options=GeminiConfig.AVAILABLE_MODELS,
            index=GeminiConfig.AVAILABLE_MODELS.index(AgentConfig.MODEL_NAME) 
                  if AgentConfig.MODEL_NAME in GeminiConfig.AVAILABLE_MODELS else 0
        )
        
        # コレクション一覧の取得
        all_collections = get_available_collections_from_qdrant()
        
        if not all_collections:
            st.warning("利用可能なコレクションが見つかりません。Qdrantサーバーを確認してください。")
            all_collections = ["(None)"]
        
        # 検索対象コレクションの選択（マルチセレクトに変更）
        selected_collections = st.multiselect(
            "検索対象コレクション (Target Collections)",
            options=all_collections,
            default=all_collections if all_collections != ["(None)"] else [], # デフォルトは全て選択
            help="エージェントが検索ツールを使用する際に、候補として提示されるコレクションです。"
        )
        
        if st.button("🗑️ 会話履歴をクリア"):
            st.session_state.chat_history = []
            st.session_state.chat_session = None
            # current_collections もクリアして再初期化を強制
            if "current_collections" in st.session_state:
                del st.session_state["current_collections"]
            # current_model もクリア
            if "current_model" in st.session_state:
                del st.session_state["current_model"]
            st.rerun()

    # 2. セッション状態の初期化と更新チェック
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # 前回のコレクション選択状態・モデルと比較
    current_collections_key = "current_collections"
    current_model_key = "current_model"
    should_reinitialize = False
    
    # selected_collections はリストなのでソートして比較
    if current_collections_key not in st.session_state:
        should_reinitialize = True
    elif sorted(st.session_state[current_collections_key]) != sorted(selected_collections):
        should_reinitialize = True
        # 設定が変わったので履歴クリアするか確認（今回はしないが、メッセージ出すなどあり）
        st.toast("検索対象コレクションが変更されたため、エージェントを再設定します。")
        
    # モデルの変更チェック
    if current_model_key not in st.session_state:
        should_reinitialize = True
    elif st.session_state[current_model_key] != selected_model:
        should_reinitialize = True
        st.toast(f"モデルが変更されました: {selected_model}")

    if should_reinitialize or "chat_session" not in st.session_state or st.session_state.chat_session is None:
        try:
            st.session_state.chat_session = setup_agent(selected_collections, selected_model)
            st.session_state[current_collections_key] = selected_collections
            st.session_state[current_model_key] = selected_model
            st.toast("エージェントの準備が完了しました。")
        except Exception as e:
            st.error(f"エージェントの初期化に失敗しました: {e}")
            return

    # 3. チャット履歴の表示
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. ユーザー入力処理
    if prompt := st.chat_input("質問を入力してください..."):
        # ユーザーのメッセージを表示
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # エージェントの応答生成
        with st.chat_message("assistant"):
            try:
                # エージェント実行（思考プロセスは内部でexpander表示）
                response_text = run_agent_turn(st.session_state.chat_session, prompt)
                
                if response_text:
                    st.markdown(response_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                else:
                    st.warning("エージェントからの応答がありませんでした。")
                    
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                logger.error(f"Chat Error: {e}", exc_info=True)