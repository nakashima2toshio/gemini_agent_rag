# agent_main.py

import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from dotenv import load_dotenv
import logging
import datetime
from config import AgentConfig, PathConfig

# Import the tool
from agent_tools import search_rag_knowledge_base, list_rag_collections

# Load environment variables
load_dotenv()

# Define System Instruction for the Agent
SYSTEM_INSTRUCTION = """
あなたは、社内ドキュメント検索システムと連携した「ハイブリッド・ナレッジ・エージェント」です。
あなたの役割は、ユーザーの質問に対して、一般的な知識と、提供されたツール（社内ナレッジ検索）を適切に使い分けて回答することです。

## 思考プロセス (Chain of Thought) の可視化

回答やツール使用の前に、必ずあなたの思考プロセスを出力してください。
**特に、なぜその行動（検索する、あるいは検索しない）を選んだのか、その理由を簡潔に述べてください。**
形式: `Thought: ここに思考を記述...`

## 行動指針 (Router Guidelines)

1.  **専門知識の検索**:
    *   ユーザーが「仕様」「設定」「エラー」「社内規定」「Wikipediaの知識」など、外部知識が必要な具体的な質問をした場合は、必ず `search_rag_knowledge_base` ツールを使用してください。
    *   **ツールの利用時には、必要に応じて `collection_name` 引数に、検索対象のQdrantコレクション名を指定してください。利用可能なコレクションが不明な場合は、`list_rag_collections` ツールを使用して一覧を取得してください。**
    *   あなたの事前学習知識だけで回答せず、必ずツールからの情報を優先してください。

2.  **一般的な会話**:
    *   挨拶、雑談、単純な計算、一般的なプログラミングの文法質問などは、ツールを使わずに直接回答してください。

3.  **正直さと不足情報の処理 (Critical)**:
    *   ツールを使用し、その結果（Observation）が「検索結果が見つかりませんでした」または関連情報を含まない場合、**絶対に**あなたの事前学習知識で捏造してはいけません。
    *   「申し訳ありませんが、提供された情報源の中には、その質問に対する回答が見つかりませんでした。」と正直に伝えてください。
    *   その上で、「もしよろしければ、もう少し詳しいキーワードや別の表現で質問していただけますか？」とユーザーを誘導してください。

4.  **回答のスタイル**:
    *   丁寧な日本語（です・ます調）で回答してください。
    *   検索結果に基づく回答の場合、「社内ナレッジによると...」や「検索結果によると...」と出典を明示すると信頼性が高まります。
"""

# Tool Map for manual execution
tools_map = {
    'search_rag_knowledge_base': search_rag_knowledge_base,
    'list_rag_collections': list_rag_collections
}

# New: Logging Setup
def setup_logging():
    log_file_path = PathConfig.LOG_DIR / AgentConfig.CHAT_LOG_FILE_NAME
    PathConfig.ensure_dirs() # Ensure log directory exists

    logging.basicConfig(
        level=getattr(logging, AgentConfig.CHAT_LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            # Optionally, add StreamHandler for console output if needed
            # logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Logger instance
logger = setup_logging()

def setup_agent():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables.")
        
    genai.configure(api_key=api_key)
    
    # Define tools
    tools_list = [search_rag_knowledge_base, list_rag_collections]
    
    # Initialize Model with Tools and System Instruction
    model = genai.GenerativeModel(
        model_name=AgentConfig.MODEL_NAME,
        tools=tools_list,
        system_instruction=SYSTEM_INSTRUCTION
    )
    
    # Start Chat with manual function calling (we will handle the loop)
    chat = model.start_chat(enable_automatic_function_calling=False)
    return chat

def print_colored(text, color="white"):
    colors = {
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def main():
    print("🤖 Hybrid Knowledge Agent (ReAct + CoT) Started!")
    print("------------------------------------------------")
    print("一般的な質問と専門知識（RAG）を自律的に使い分け、思考プロセスを表示します。")
    print("終了するには 'exit' または 'quit' と入力してください。\n")
    
    logger.info(f"Agent session started at {datetime.datetime.now()}")

    try:
        chat = setup_agent()
    except Exception as e:
        print(f"Error setting up agent: {e}")
        logger.error(f"Error setting up agent: {e}")
        return

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                logger.info("User requested exit. Agent session ended.")
                print("Agent: Goodbye!")
                break
            
            logger.info(f"User Input: {user_input}")
            # Removed redundant print_colored("You: ...") to prevent duplication
            
            # Initial message
            response = chat.send_message(user_input)
            
            # ReAct Loop
            while True:
                function_call_found = False
                
                # Check all parts
                for part in response.parts:
                    # 1. Handle Text (Thought or Answer)
                    if part.text:
                        log_message = part.text.strip()
                        if "Thought:" in log_message or "考え:" in log_message:
                             print_colored(f"\n[🧠 Thought]\n{log_message}", "cyan")
                             logger.info(f"Agent Thought: {log_message}")
                        else:
                             print(f"\nAgent: {log_message}")
                             logger.info(f"Agent Response: {log_message}")

                    # 2. Handle Function Call
                    if part.function_call:
                        function_call_found = True
                        fn = part.function_call
                        tool_name = fn.name
                        tool_args = dict(fn.args)
                        
                        print_colored(f"\n[🛠️ Tool Call] {tool_name}({tool_args})", "yellow")
                        logger.info(f"Agent Tool Call: {tool_name}({tool_args})")
                        
                        # Execute Tool
                        if tool_name in tools_map:
                            try:
                                tool_result = tools_map[tool_name](**tool_args)
                                log_tool_result = str(tool_result)[:500] + "..." if len(str(tool_result)) > 500 else str(tool_result)
                                logger.info(f"Tool Result: {log_tool_result}")
                            except Exception as tool_err:
                                tool_result = f"Error executing tool: {tool_err}"
                                logger.error(f"Error executing tool '{tool_name}': {tool_err}")
                        else:
                            tool_result = f"Error: Tool '{tool_name}' not found."
                            logger.warning(f"Attempted to call unknown tool: {tool_name}")
                        
                        # Send Result back to Model
                        response = chat.send_message(
                            genai.protos.Content(
                                parts=[genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=tool_name,
                                        response={'result': tool_result}
                                    )
                                )]
                            )
                        )
                        break 

                if function_call_found:
                    continue
                else:
                    break
            
        except KeyboardInterrupt:
            logger.info("User interrupted with Ctrl+C. Agent session ended.")
            print("\nAgent: Goodbye!")
            break
        except Exception as e:
            print(f"\nError during chat: {e}")
            logger.error(f"Error during chat session: {e}", exc_info=True)
            continue

if __name__ == "__main__":
    main()
