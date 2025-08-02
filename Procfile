release: mkdir -p faiss_indices && echo "{}" > sessions.json
web: gunicorn --worker-class gevent --bind 0.0.0.0:$PORT app:app
