from sqlalchemy.ext.asyncio import create_async_engine
# from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
import os
from urllib.parse import quote_plus

# Load environment variables from .env
load_dotenv()

# Fetch variables
USER = os.getenv("user")
PASSWORD = quote_plus(os.getenv("password"))
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

# Construct the SQLAlchemy connection string
DATABASE_URL = f"postgresql+asyncpg://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"

# Create the SQLAlchemy engine


engine = create_async_engine(DATABASE_URL, connect_args={"ssl": "require"})
# If using Transaction Pooler or Session Pooler, we want to ensure we disable SQLAlchemy client side pooling -
# https://docs.sqlalchemy.org/en/20/core/pooling.html#switching-pool-implementations
# engine = create_engine(DATABASE_URL, poolclass=NullPool)

# Test the connection
