import os

from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv

# load_dotenv()
load_dotenv(".env.webhook")

app = FastAPI(title="WhatsApp Webhook")

VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")

if not VERIFY_TOKEN:
    raise RuntimeError(
        "META_VERIFY_TOKEN is missing from your .env file"
    )

# --------------------------------------------------
# GET /webhook
# Meta uses this endpoint to verify the webhook
# --------------------------------------------------

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_verify_token: str | None = Query(
        default=None,
        alias="hub.verify_token"
    ),
    hub_challenge: str | None = Query(
        default=None,
        alias="hub.challenge"
    ),
):
    print("\n========== WEBHOOK VERIFICATION ==========")
    print("hub.mode:", hub_mode)
    print("hub.verify_token:", hub_verify_token)
    print("hub.challenge:", hub_challenge)

    # Meta sends mode=subscribe
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:

        print("✅ WEBHOOK VERIFIED")

        # IMPORTANT:
        # Meta expects the challenge as plain text
        return PlainTextResponse(
            content=hub_challenge or "",
            status_code=200,
        )

    print("❌ WEBHOOK VERIFICATION FAILED")

    return PlainTextResponse(
        content="Forbidden",
        status_code=403,
    )


# --------------------------------------------------
# POST /webhook
# Meta sends WhatsApp messages/events here
# --------------------------------------------------

@app.post("/webhook")
async def receive_webhook(request: Request):

    print("\n========== WHATSAPP WEBHOOK ==========")

    try:
        data = await request.json()

        print("Incoming data:")
        print(data)

        # TODO:
        # Process WhatsApp messages here

        return JSONResponse(
            content={"status": "received"},
            status_code=200,
        )

    except Exception as e:

        print("❌ Webhook error:", e)

        return JSONResponse(
            content={
                "status": "error",
                "message": str(e),
            },
            status_code=200,
        )


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "online",
        "webhook": "/webhook",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }