"""FastAPI application for the Algorithmic Commons platform API."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialise database pool and graph driver on startup."""
    db_pool = None
    graph_driver = None

    # PostgreSQL
    postgres_uri = os.environ.get("AGEOM_POSTGRES_URI", "")
    if postgres_uri:
        try:
            import asyncpg

            db_pool = await asyncpg.create_pool(
                postgres_uri,
                min_size=2,
                max_size=10,
                statement_cache_size=0,  # Supabase PgBouncer compat
            )
            app.state.db_pool = db_pool
        except Exception:
            db_pool = None

    # Memgraph
    memgraph_uri = os.environ.get("AGEOM_MEMGRAPH_URI", "")
    if memgraph_uri:
        try:
            from neo4j import AsyncGraphDatabase

            graph_driver = AsyncGraphDatabase.driver(memgraph_uri, auth=None)
            app.state.graph_driver = graph_driver
        except Exception:
            graph_driver = None

    yield

    if graph_driver is not None:
        try:
            await graph_driver.close()
        except Exception:
            pass
    if db_pool is not None:
        try:
            await db_pool.close()
        except Exception:
            pass


def create_app() -> FastAPI:
    """Factory for the platform API."""
    application = FastAPI(
        title="Algorithmic Commons API",
        version="0.1.0",
        lifespan=_lifespan,
    )

    from ageom.api.routers.auth import router as auth_router
    from ageom.api.routers.bounty import router as bounty_router
    from ageom.api.routers.catalog import router as catalog_router
    from ageom.api.routers.registry import router as registry_router

    application.include_router(auth_router, tags=["auth"])
    application.include_router(registry_router, prefix="/atoms", tags=["registry"])
    application.include_router(bounty_router, prefix="/bounties", tags=["bounties"])
    application.include_router(catalog_router, prefix="/catalog", tags=["catalog"])

    return application


app = create_app()
