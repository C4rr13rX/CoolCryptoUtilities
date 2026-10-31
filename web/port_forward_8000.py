import asyncio
import sys

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        remote_reader, remote_writer = await asyncio.open_connection('127.0.0.1', 8001)
    except Exception:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return

    async def forward(src: asyncio.StreamReader, dst: asyncio.StreamWriter):
        try:
            while True:
                data = await src.read(65536)
                if not data:
                    break
                dst.write(data)
                await dst.drain()
        except Exception:
            pass
        finally:
            try:
                dst.close()
            except Exception:
                pass

    await asyncio.gather(
        forward(reader, remote_writer),
        forward(remote_reader, writer),
        return_exceptions=True
    )

async def main():
    server = await asyncio.start_server(handle_client, '127.0.0.1', 8000, reuse_address=True)
    print("Port forwarder active: http://127.0.0.1:8000 -> http://127.0.0.1:8001", flush=True)
    async with server:
        await server.serve_forever()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
