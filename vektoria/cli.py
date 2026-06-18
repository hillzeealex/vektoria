"""Command-line entry point: `vektoria serve` launches the REST API."""

import argparse
import os


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="vektoria", description="Vektoria vector database")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the REST API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data-dir", default=os.environ.get("VK_DATA_DIR", "./data"))

    args = parser.parse_args(argv)

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError:
            raise SystemExit(
                "The server needs the 'server' extra. Install with: pip install 'vektoria[server]'"
            )
        os.environ.setdefault("VK_DATA_DIR", args.data_dir)
        uvicorn.run("vektoria.server:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
