from dotenv import load_dotenv
import os
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import json
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)

from agent import get_agent
from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations)

from rag import add_document_to_rag
from tools import set_current_thread_id

# ── Auth ──────────────────────────────────────────────────────────────────────
from auth import (
    init_auth_db,
    get_current_user,
    get_user_by_username,
    get_user_by_email,
    create_user,
    verify_password,
    create_session_token,
    COOKIE_NAME,
    SESSION_MAX_AGE,
)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

init_db()
init_auth_db()


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.get("/login")
async def login_page(request: Request):
    # Already logged in → redirect to home
    if get_current_user(request):
        return RedirectResponse("/", status_code=302)
    success = request.query_params.get("success")
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": None, "success": success},
    )


@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    user = get_user_by_username(username.strip())
    if not user or not verify_password(password, user.password):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Invalid username or password.", "success": None},
            status_code=401,
        )

    token = create_session_token(user.username)
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=False,  # set True in production with HTTPS
    )
    return response


@app.get("/signup")
async def signup_page(request: Request):
    if get_current_user(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="signup.html",
        context={"error": None},
    )


@app.post("/signup")
async def signup_submit(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    username = username.strip()
    email = email.strip().lower()

    # Validations
    if len(username) < 3 or len(username) > 32:
        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={"error": "Username must be 3–32 characters."},
            status_code=400,
        )

    if not username.replace("_", "").isalnum():
        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={"error": "Username can only contain letters, numbers, and underscores."},
            status_code=400,
        )

    if len(password) < 6:
        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={"error": "Password must be at least 6 characters."},
            status_code=400,
        )

    if password != confirm_password:
        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={"error": "Passwords do not match."},
            status_code=400,
        )

    if get_user_by_username(username):
        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={"error": "Username already taken. Please choose another."},
            status_code=400,
        )

    if get_user_by_email(email):
        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={"error": "An account with that email already exists."},
            status_code=400,
        )

    create_user(username=username, email=email, plain_password=password)
    return RedirectResponse("/login?success=Account+created!+Please+sign+in.", status_code=302)


@app.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie(COOKIE_NAME)
    return response


# ── Protected app routes ──────────────────────────────────────────────────────

@app.get("/")
async def home(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"username": user}
    )


@app.get("/conversations")
async def conversations(request: Request):
    if not get_current_user(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    items = list_conversations()

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat()
            }
            for item in items
        ]
    }



@app.get("/history/{thread_id}")
async def history(thread_id: str, request: Request):
    if not get_current_user(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    messages = get_chat_history(thread_id)

    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    }




@app.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    thread_id: str = Form(...),
):
    if not get_current_user(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        allowed_extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".csv"]

        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV."
                },
                status_code=400
            )

        file_id = str(uuid.uuid4())
        safe_filename = filename.replace(" ", "_")
        file_path = f"uploads/{file_id}_{safe_filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        create_or_update_conversation(thread_id, "Uploaded document")

        result = add_document_to_rag(
            file_path=file_path,
            thread_id=thread_id
        )

        return JSONResponse({
            "success": True,
            "message": f"Uploaded {result['filename']} and created {result['chunks']} chunks."
        })

    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "message": str(e)
            },
            status_code=500
        )



def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def should_stream_chunk(chunk, metadata) -> bool:
    """
    This prevents raw tool/search/RAG JSON from appearing in the frontend.

    We only stream normal AI text chunks.
    We do NOT stream:
    - ToolMessage
    - messages from tool nodes
    - tool call chunks
    - raw tool outputs
    """

    metadata = metadata or {}

    node_name = str(metadata.get("langgraph_node", "")).lower()

    if "tool" in node_name:
        return False

    if isinstance(chunk, ToolMessage):
        return False

    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False

    if getattr(chunk, "tool_calls", None):
        return False

    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}

    if additional_kwargs.get("tool_calls"):
        return False

    return True


def extract_text_from_chunk(chunk) -> str:
    content = getattr(chunk, "content", "")

    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])

        return "".join(text_parts)

    return ""



@app.post("/chat/stream")
async def chat_stream(request: Request):
    if not get_current_user(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body."},
            status_code=400
        )

    user_message = data.get("message", "")
    thread_id = data.get("thread_id", "default")
    selected_model = data.get("model", "gemini-3.5-flash")

    if not user_message.strip():
        return JSONResponse(
            {"error": "Message is required."},
            status_code=400
        )

    agent = get_agent(selected_model)

    create_or_update_conversation(thread_id, user_message)
    save_chat_message(thread_id, "user", user_message)

    set_current_thread_id(thread_id)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    def event_generator():
        final_answer = ""

        try:
            inputs = {
                "messages": [
                    HumanMessage(content=user_message)
                ]
            }

            for chunk, metadata in agent.stream(
                inputs,
                config=config,
                stream_mode="messages"
            ):
                if not should_stream_chunk(chunk, metadata):
                    continue

                token = extract_text_from_chunk(chunk)

                if token:
                    final_answer += token
                    yield sse_data({"token": token})

            if final_answer.strip():
                save_chat_message(thread_id, "assistant", final_answer)

            yield sse_data({"done": True})

        except Exception as e:
            yield sse_data({"error": str(e)})
            yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )




if __name__ == "__main__":
   
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
