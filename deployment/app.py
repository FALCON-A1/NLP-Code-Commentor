"""
Web Application - FastAPI
==========================
REST API and web interface for the CodeCommentator tool.
"""

import os
import sys
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="CodeCommentator NLP",
    description="Automated Code Documentation Generator",
    version="1.0.0"
)

# Setup templates and static files
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class CodeInput(BaseModel):
    code: str
    model_type: str = "codebert"


class DocumentationOutput(BaseModel):
    documentation: str
    model_used: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/generate", response_model=DocumentationOutput)
async def generate_documentation(input_data: CodeInput):
    """
    API endpoint to generate documentation for code.

    Args:
        input_data: CodeInput with source code and model type.

    Returns:
        Generated documentation.
    """
    from deployment.inference import get_generator

    generator = get_generator(input_data.model_type)
    doc = generator.generate(input_data.code)

    return DocumentationOutput(
        documentation=doc,
        model_used=input_data.model_type,
    )


@app.post("/generate", response_class=HTMLResponse)
async def generate_form(request: Request, code: str = Form(...),
                        model_type: str = Form("codebert")):
    """Handle form submission from web UI."""
    from deployment.inference import get_generator

    generator = get_generator(model_type)
    doc = generator.generate(code)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "code": code,
        "documentation": doc,
        "model_type": model_type,
    })


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
