from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .core.config import WEB_DIR, MODEL_PATH
from .ml.loader import load_model
from .api.routes import router

def create_app() -> FastAPI:
    app = FastAPI()

    # статика + index
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def root():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/favicon.ico")
    def favicon():
        return {}

    # CORS (можно убрать, если фронт всегда с этого же домена/порта)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # роуты API
    app.include_router(router)

    # загрузка модели в app.state
    @app.on_event("startup")
    def on_startup():
        app.state.model_bundle = load_model(MODEL_PATH)
        b = app.state.model_bundle
        print(f"[OK] Loaded model: {b.model_path} | mean={b.mean} std={b.std}")

    return app

app = create_app()