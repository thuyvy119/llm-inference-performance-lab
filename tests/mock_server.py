from __future__ import annotations
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))

        self.rfile.read(content_length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant"
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "content": "GPU"
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "content": " is faster."
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                },
            },
        ]

        for chunk in chunks:
            data = json.dumps(chunk)

            self.wfile.write(
                f"data: {data}\n\n".encode("utf-8")
            )
            self.wfile.flush()
            time.sleep(0.01)

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args):
        """Suppress default HTTP server logging."""


def run_server() -> None:
    """Start the mock server."""
    server = HTTPServer(
        ("127.0.0.1", 18000),
        MockHandler,
    )

    print("Mock server listening on http://127.0.0.1:18000")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping mock server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()