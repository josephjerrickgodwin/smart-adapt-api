class StreamingService:
    @staticmethod
    async def stream(data: str):
        yield f'data: {data}\n\n' if data else 'data: [END]'


streaming_service = StreamingService()
