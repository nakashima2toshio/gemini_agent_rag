from agent_main import setup_agent, tools_map
import google.generativeai as genai
import time

def process_turn(chat, user_input):
    print(f"User: {user_input}")
    response = chat.send_message(user_input)
    
    while True:
        # Debug: Print all parts structure
        # print(f"DEBUG: Parts count: {len(response.parts)}")
        
        function_call_found = False
        
        for part in response.parts:
            if part.text:
                print(f"Agent (Text): {part.text.strip()}")
            
            if part.function_call:
                function_call_found = True
                fn = part.function_call
                print(f"Agent (Tool Call): {fn.name}({fn.args})")
                
                # Execute Tool
                if fn.name in tools_map:
                    print(f"[System] Executing {fn.name}...")
                    result = tools_map[fn.name](**fn.args)
                else:
                    result = "Error: Tool not found"
                
                # Send result back
                response = chat.send_message(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=fn.name,
                                response={'result': result}
                            )
                        )]
                    )
                )
                break # Break inner for-loop to process next response
        
        if function_call_found:
            continue # Continue while loop
        else:
            break # No function call in this turn -> Finished

def test():
    print("--- Test 1: General Chat ---")
    try:
        chat = setup_agent()
        process_turn(chat, "こんにちは、元気ですか？")
    except Exception as e:
        print(f"Test 1 Failed: {e}")
    
    print("\n--- Test 2: Knowledge Search (RAG) ---")
    try:
        chat = setup_agent() # Reset chat
        process_turn(chat, "東京タワーの高さは何メートルですか？")
    except Exception as e:
        print(f"Test 2 Failed: {e}")

if __name__ == "__main__":
    test()