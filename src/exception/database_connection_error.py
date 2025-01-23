from src.exception.database_error import DatabaseError


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""
    pass
