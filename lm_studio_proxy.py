from fastapi import FastAPI, Request
import httpx
import uvicorn

app = FastAPI()

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str):
    url = f"http://127.0.0.1:1234/{path}"
    
    # Exclude host header 
    headers = dict(request.headers)
    headers.pop("host", None)

    async with httpx.AsyncClient() as client:
        req = client.build_request(
            request.method,
            url,
            headers=headers,
            content=await request.body()
        )
        response = await client.send(req, stream=True)
        return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1235)
