import os
from app.app import app

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    app.run(host=host, debug=app.config["DEBUG"], load_dotenv=False)
    