class StreamingService:
    @staticmethod
    def stream(data: str, stop_stream: bool = False):
        if data:
            yield f'data: {data}\n\n'
        if stop_stream:
            yield f'data: [END]\n\n'


streaming_service = StreamingService()
