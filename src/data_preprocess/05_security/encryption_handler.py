from cryptography.fernet import Fernet
import os


class EncryptionHandler:
    """
    Handles encryption and decryption for data at rest and in transit.
    """

    def __init__(self, key_path: str = "encryption_key.key"):
        """
        Initialize the encryption handler.

        Args:
            key_path: Path to the encryption key file.
        """
        self.key_path = key_path
        self.key = self._load_or_generate_key()

    def _load_or_generate_key(self) -> bytes:
        """
        Load an existing encryption key or generate a new one.

        Returns:
            The encryption key as bytes.
        """
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as key_file:
                return key_file.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_path, "wb") as key_file:
                key_file.write(key)
            return key

    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt data.

        Args:
            data: The data to encrypt as bytes.

        Returns:
            The encrypted data as bytes.
        """
        fernet = Fernet(self.key)
        return fernet.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data.

        Args:
            encrypted_data: The encrypted data as bytes.

        Returns:
            The decrypted data as bytes.
        """
        fernet = Fernet(self.key)
        return fernet.decrypt(encrypted_data)

    def encrypt_file(self, file_path: str):
        """
        Encrypt a file (data at rest).

        Args:
            file_path: Path to the file to encrypt.
        """
        with open(file_path, "rb") as file:
            data = file.read()
        encrypted_data = self.encrypt_data(data)
        with open(file_path, "wb") as file:
            file.write(encrypted_data)

    def decrypt_file(self, file_path: str):
        """
        Decrypt a file (data at rest).

        Args:
            file_path: Path to the file to decrypt.
        """
        with open(file_path, "rb") as file:
            encrypted_data = file.read()
        data = self.decrypt_data(encrypted_data)
        with open(file_path, "wb") as file:
            file.write(data)
