from src.exception.database_error import DatabaseError


class DuplicateEmailError(DatabaseError):
    """Raised when attempting to create a record with existing email"""
    pass
