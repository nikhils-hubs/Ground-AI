from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def send():
    return {
        "hello world"
    }