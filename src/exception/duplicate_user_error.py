from src.exception.database_error import DatabaseError


class DuplicateUserError(DatabaseError):
    """Raised when attempting to create a record with existing user_id"""
    pass
