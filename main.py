import os
import requests
from fastapi import FastAPI, Request, Response, Query
from dotenv import load_dotenv

# Import your compiled LangGraph application instance
from conditional_RAG import app as graph

load_dotenv()

app = FastAPI()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "my_secret_token")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

# 1. GET Endpoint for Meta Webhook Handshake Verification
@app.get("/webhook")
async def verify_webhook(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge")
):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return Response(content=challenge, media_type="text/plain", status_code=200)
    return Response(content="Verification failed", status_code=403)

# 2. POST Endpoint to Receive WhatsApp Messages and Invoke LangGraph
@app.post("/webhook")
async def handle_whatsapp_message(request: Request):
    payload = await request.json()

    try:
        entries = payload.get("entry", [])
        for entry in entries:
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])

                if messages:
                    msg = messages[0]
                    sender_number = msg.get("from")  # e.g., "923105139377"
                    
                    if msg.get("type") == "text":
                        user_text = msg.get("text", {}).get("body")
                        print(f"Received message: '{user_text}' from {sender_number}")
                        
                        # Trigger LangGraph Execution
                        config = {"configurable": {"thread_id": sender_number}}
                        response_state = graph.invoke(
                            {"messages": [("user", user_text)]}, 
                            config=config
                        )
                        
                        # Extract final text output from LangGraph
                        bot_reply = response_state["messages"][-1].content
                        
                        # Send the AI response back to WhatsApp
                        send_whatsapp_message(sender_number, bot_reply)

    except Exception as e:
        print(f"Error handling webhook payload: {e}")

    return Response(content="EVENT_RECEIVED", status_code=200)

def send_whatsapp_message(recipient_number: str, text_content: str):
    url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": recipient_number,
        "type": "text",
        "text": {"body": text_content}
    }
    res = requests.post(url, json=data, headers=headers)
    print("WhatsApp Response:", res.status_code, res.json())