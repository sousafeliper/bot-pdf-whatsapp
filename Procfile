release: mkdir -p faiss_indices && touch sessions.json
web: gunicorn --worker-class gevent -b 0.0.0.0:$PORT app:app
