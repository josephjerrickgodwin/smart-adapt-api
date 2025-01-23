from src.exception import DatabaseError


class InvalidDataError(DatabaseError):
    """Raised when the data is invalid or corrupted"""
    pass
